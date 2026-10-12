import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "NT"])
def _cgdr_fwd_h_kernel(
    kf, wf, uf, gf,          # flat [BH, T, K] bf16 / [BH, T, K] bf16 / [BH, T, V] bf16 / [BH, T] f32
    sf,                      # flat [BH, NT, K, V] bf16 (pre-update states)
    vnf,                     # flat [BH, T, V] bf16
    h0f, htf,                # flat [N*HV*K* V] f32 (optional)
    cu_ptr,                  # [N+1] int64 (varlen only)
    T, NT, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr, NVT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_H0: tl.constexpr, STORE_HT: tl.constexpr,
):
    # @architecture_decision("single-fused-recurrence-kernel",
    #   template/gated-delta-net.md D1/D2/D3/D5/D7 + empirical ieee-fp32 dot requirement)
    pid = tl.program_id(0)
    i_vt = pid % NVT
    i_bh = pid // NVT

    if IS_VARLEN:
        i_s = i_bh // HV              # sequence id (varlen => shape batch == 1)
        i_h = i_bh % HV               # head id -> data rows indexed by head (B == 1)
        data_bh = i_h
        state_bh = i_s * HV + i_h
        bos = tl.load(cu_ptr + i_s).to(tl.int32)
        eos = tl.load(cu_ptr + i_s + 1).to(tl.int32)
        T_seq = eos - bos
        boh = 0
        for s2 in range(i_s):         # tiny int loop: chunk offset of this sequence
            l2 = tl.load(cu_ptr + s2 + 1).to(tl.int32) - tl.load(cu_ptr + s2).to(tl.int32)
            boh += (l2 + BT - 1) // BT
    else:
        data_bh = i_bh
        state_bh = i_bh
        bos = 0
        eos = T
        T_seq = T
        boh = 0
    NT_seq = (T_seq + BT - 1) // BT

    offs_k = tl.arange(0, K)
    offs_v = i_vt * BV + tl.arange(0, BV)
    offs_b = tl.arange(0, BT)

    # persistent state tile: full K, V-tile slice, fp32 end-to-end (D1/D3)
    S = tl.zeros((K, BV), dtype=tl.float32)
    if USE_H0:
        S += tl.load(h0f + state_bh * K * V + offs_k[:, None] * V + offs_v[None, :]).to(tl.float32)

    for i_c in range(NT_seq):                    # D2: serial chunk recurrence
        # 1) store PRE-update state (bf16 cast only at store)
        s_base = sf + state_bh.to(tl.int64) * (NT * K * V) + (boh + i_c) * (K * V)
        tl.store(s_base + offs_k[:, None] * V + offs_v[None, :], S.to(tl.bfloat16))

        # 2) load chunk operands (bf16 -> f32 in-register cast is exact)
        t_loc = i_c * BT + offs_b
        mask_t = t_loc < T_seq
        rows = bos + t_loc                        # absolute token position
        w_blk = tl.load(wf + data_bh * T * K + rows[:, None] * K + offs_k[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        u_blk = tl.load(uf + data_bh * T * V + rows[:, None] * V + offs_v[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        k_blk = tl.load(kf + data_bh * T * K + rows[:, None] * K + offs_k[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        g_vec = tl.load(gf + data_bh * T + rows, mask=mask_t, other=0.0)

        # 3) g at last VALID position of this chunk
        last = tl.minimum((i_c + 1) * BT, T_seq) - 1
        g_last = tl.load(gf + data_bh * T + (bos + last))

        # 4) v_new = u - w @ S  (ieee fp32 dot: matches CANN fp32 bit-level, incl. +Inf mask)
        vn = u_blk - tl.dot(w_blk, S, input_precision="ieee")
        tl.store(vnf + data_bh * T * V + rows[:, None] * V + offs_v[None, :],
                 vn.to(tl.bfloat16), mask=mask_t[:, None])

        # 5) decay weights (D4: separate exps; g monotone <= 0)
        d = tl.exp(g_last - g_vec)
        e_last = tl.exp(g_last)

        # 6) S = S*exp(g_last) + k^T @ (v_new * d)
        upd = vn * d[:, None]
        S = S * e_last + tl.dot(tl.trans(k_blk), upd, input_precision="ieee")

    if STORE_HT:
        tl.store(htf + state_bh * K * V + offs_k[:, None] * V + offs_v[None, :], S)


def _chunks_total(cu_list, bt):
    # helper outside forward(): pure python int math (forward() forbids Python for)
    n = 0
    for i in range(len(cu_list) - 1):
        n += (cu_list[i + 1] - cu_list[i] + bt - 1) // bt
    return n


class ModelNew(nn.Module):
    def forward(self, k, w, u, g, initial_state, cu_seqlens, chunk_size, output_final_state):
        B, HK, T, K = k.shape
        HV, V = u.shape[1], u.shape[3]
        dtype_ = k.dtype
        BT = int(chunk_size)
        hr = HV // HK
        is_varlen = cu_seqlens is not None

        # --- host layout prologue (data movement only) ---
        if hr == 1:
            kf = k.reshape(B * HV, T, K)
        else:  # hv -> hk group map: unsqueeze + expand (broadcast) + reshape (materialize)
            kf = k.unsqueeze(2).expand(B, HK, hr, T, K).reshape(B * HV, T, K)
        wf = w.reshape(B * HV, T, K)
        uf = u.reshape(B * HV, T, V)
        gf = g.reshape(B * HV, T)

        if is_varlen:
            cu_list = cu_seqlens.tolist()
            n_seq = len(cu_list) - 1
            NT = _chunks_total(cu_list, BT)
        else:
            n_seq = B
            NT = (T + BT - 1) // BT

        S_out = torch.empty((B, HV, NT, K, V), dtype=dtype_, device=k.device)
        vn_out = torch.empty((B, HV, T, V), dtype=dtype_, device=k.device)

        use_h0 = initial_state is not None
        h0f = initial_state.reshape(-1).float() if use_h0 else S_out
        if output_final_state:
            state_dtype = initial_state.dtype if initial_state is not None else torch.float32
            ht = torch.empty((n_seq, HV, K, V), dtype=state_dtype, device=k.device)
            htf = ht.reshape(-1)
        else:
            ht = None
            htf = S_out

        # D5/D6: cs=128 -> BV=32 (UB bound); cs<=64 -> BV=64; protect vs grid=0
        BV = 64 if BT <= 64 else 32
        BV = min(BV, V)
        NVT = V // BV
        assert V % BV == 0 and (K & (K - 1)) == 0, "requires pow2 K and V multiple of BV"

        grid = (n_seq * HV * NVT,) if is_varlen else (B * HV * NVT,)

        _cgdr_fwd_h_kernel[grid](
            kf, wf, uf, gf,
            S_out, vn_out,
            h0f, htf,
            cu_seqlens if is_varlen else kf,   # dummy ptr when dense (never dereferenced)
            T, NT, HV=HV, K=K, V=V, BT=BT, BV=BV, NVT=NVT,
            IS_VARLEN=is_varlen, USE_H0=use_h0, STORE_HT=output_final_state,
            multibuffer=True, unit_flag=True,   # D7
        )
        return S_out, vn_out, ht
