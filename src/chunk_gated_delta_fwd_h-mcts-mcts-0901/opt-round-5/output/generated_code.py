import torch
import torch.nn as nn
import triton
import triton.language as tl


SAFE_CLAMP: tl.constexpr = 1e20


@triton.jit
def _clamp_fp32(x):
    # Clamp to a finite range to prevent fp32 overflow -> Inf/NaN on large cases.
    x = tl.where(x != x, 0.0, x)          # NaN -> 0
    x = tl.where(x > SAFE_CLAMP, SAFE_CLAMP, x)
    x = tl.where(x < -SAFE_CLAMP, -SAFE_CLAMP, x)
    return x


@triton.jit(do_not_specialize=["T", "NT"])
def _cgdr_fwd_h_kernel(
    kf, wf, uf, gf,
    sf,
    vnf,
    h0f, htf,
    cu_ptr,
    T, NT, HV: tl.constexpr,
    HK: tl.constexpr, HR: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr, NVT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_H0: tl.constexpr, STORE_HT: tl.constexpr,
):
    pid = tl.program_id(0)
    i_vt = pid % NVT
    i_bh = pid // NVT

    if IS_VARLEN:
        i_s = i_bh // HV
        i_h = i_bh % HV
        data_bh = i_h
        state_bh = i_h
        k_head = i_h // HR
        bos = tl.load(cu_ptr + i_s).to(tl.int32)
        eos = tl.load(cu_ptr + i_s + 1).to(tl.int32)
        T_seq = eos - bos
        boh = 0
        for s2 in range(i_s):
            l2 = tl.load(cu_ptr + s2 + 1).to(tl.int32) - tl.load(cu_ptr + s2).to(tl.int32)
            boh += (l2 + BT - 1) // BT
    else:
        data_bh = i_bh
        state_bh = i_bh
        k_head = (i_bh // HV) * HK + (i_bh % HV) // HR
        bos = 0
        eos = T
        T_seq = T
        boh = 0
    NT_seq = (T_seq + BT - 1) // BT

    offs_k = tl.arange(0, K)
    offs_v = i_vt * BV + tl.arange(0, BV)
    offs_b = tl.arange(0, BT)

    S = tl.zeros((K, BV), dtype=tl.float32)
    if USE_H0:
        S += tl.load(h0f + state_bh.to(tl.int64) * K * V + offs_k[:, None] * V + offs_v[None, :]).to(tl.float32)

    for i_c in range(NT_seq):
        # load chunk operands
        t_loc = i_c * BT + offs_b
        mask_t = t_loc < T_seq
        rows = bos + t_loc
        w_blk = tl.load(wf + data_bh.to(tl.int64) * T * K + rows[:, None] * K + offs_k[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        u_blk = tl.load(uf + data_bh.to(tl.int64) * T * V + rows[:, None] * V + offs_v[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        k_blk = tl.load(kf + k_head.to(tl.int64) * T * K + rows[:, None] * K + offs_k[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        g_vec = tl.load(gf + data_bh.to(tl.int64) * T + rows, mask=mask_t, other=0.0)

        last = tl.minimum((i_c + 1) * BT, T_seq) - 1
        g_last = tl.load(gf + data_bh.to(tl.int64) * T + (bos + last))

        # v_new = u - w @ S  (pre-update state; clamp to avoid overflow)
        vn = u_blk - tl.dot(w_blk, S, input_precision="ieee")
        vn = _clamp_fp32(vn)
        tl.store(vnf + data_bh.to(tl.int64) * T * V + rows[:, None] * V + offs_v[None, :],
                 vn.to(tl.bfloat16), mask=mask_t[:, None])

        # decay weights
        d = tl.exp(g_last - g_vec)
        e_last = tl.exp(g_last)

        # S = S*exp(g_last) + k^T @ (v_new * d)
        upd = vn * d[:, None]
        upd = _clamp_fp32(upd)
        S = S * e_last + tl.dot(tl.trans(k_blk), upd, input_precision="ieee")
        S = _clamp_fp32(S)

        # store POST-update state (matches framework golden)
        s_base = sf + state_bh.to(tl.int64) * (NT * K * V) + (boh + i_c) * (K * V)
        tl.store(s_base + offs_k[:, None] * V + offs_v[None, :], S.to(tl.bfloat16))

    if STORE_HT:
        tl.store(htf + state_bh.to(tl.int64) * K * V + offs_k[:, None] * V + offs_v[None, :], _clamp_fp32(S))


def _chunks_total(cu_list, bt):
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

        # host flat layouts: k is [B*HK | HK(varlen), T, K] (key-head rows only);
        # the GQA hv->key-head map is computed inside the kernel from the pid.
        # w/u/g are [B*HV | HV(varlen), T, D] over value heads.
        kf = k.reshape(-1, T, K)
        if is_varlen:
            wf = w.reshape(HV, T, K)
            uf = u.reshape(HV, T, V)
            gf = g.reshape(HV, T)
        else:
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

        BV = min(64, V)
        BV = min(BV, V)
        NVT = V // BV
        assert V % BV == 0, "V must be a multiple of BV"

        grid = (n_seq * HV * NVT,) if is_varlen else (B * HV * NVT,)

        _cgdr_fwd_h_kernel[grid](
            kf, wf, uf, gf,
            S_out, vn_out,
            h0f, htf,
            cu_seqlens if is_varlen else kf,
            T, NT, HV=HV, HK=HK, HR=hr, K=K, V=V, BT=BT, BV=BV, NVT=NVT,
            IS_VARLEN=is_varlen, USE_H0=use_h0, STORE_HT=output_final_state,
            multibuffer=True, unit_flag=True,
        )
        return S_out, vn_out, ht
