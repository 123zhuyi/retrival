"""Triton skeleton for ChunkGatedDeltaRuleFwdH (GDN/KDA chunk 间状态递推).

P2 骨架（可运行形态），与 mapping.md 逐条对应：

- grid = (N_seq * HV * NVB,)；每 program 持有 h 的 [K, BLOCK_V] fp32 条带，
  chunk 维在 program 内串行递推（V 维列条带相互独立——比 AscendC 的 (b,h) 任务
  更细一级的并行轴，天然消解 CV 四段流水与全部 GM workspace）。
- 两个 GEMM 都在 program 内：v_new = u - w @ h；h += k^T @ v_up。
  k 用转置寻址直接 load [K, BT] tile——禁用 tl.trans（Triton-Ascend 标量化 ~15x，
  见 runbooks/compute-pipeline/triton-ascend-trans-scalarization.md）。
- g = 标量门（自然对数域，chunk 内 cumsum 语义）：decay[t]=exp(g_last-g[t]) 乘进
  v_up，h 整体乘 exp(g_last)；gk = 逐 K 门（log2 域）：h 逐行乘 exp(gk_last*ln2)。
  ⚠️ g_last/gk_last 必须取 chunk 内最后一个有效 token（actual-1 行，尾 chunk 不是 BT-1）。
- 精度对齐源码：h 条带 fp32 常驻；dot 输入 cast 到 input dtype（对齐 mmad fp16 输入、
  fp32 累加）；store 隐式 RTNE；fp32 final_state 直通天然成立。
- 后端契约：无 autotune / num_stages（不生效）；无 % 运算符（用 a-(a//b)*b）；
  mask 全 int32 比较；无 tl.trans。
"""

import torch
import triton
import triton.language as tl

LN2 = 0.6931471805599453


@triton.jit
def _chunk_gdn_fwd_h_kernel(
    k_ptr, w_ptr, u_ptr, g_ptr, gk_ptr,
    is_ptr, h_ptr, v_new_ptr, fs_ptr,
    cu_ptr, co_ptr,
    HV, G, T,
    skb, skh, skt,
    swb, swh, swt,
    sub, suh, sut,
    sgb, sgh,
    sgkb, sgkh, sgkt,
    sib, sih,
    shb, shh, shc, shk,
    svb, svh, svt,
    sfb, sfh,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr,
    USE_INITIAL: tl.constexpr, STORE_FINAL: tl.constexpr,
    VARLEN: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    NVB: tl.constexpr = V // BLOCK_V
    vb = pid - (pid // NVB) * NVB
    tmp = pid // NVB
    hv = tmp - (tmp // HV) * HV
    b = tmp // HV
    hk = hv // G

    if VARLEN:
        bos = tl.load(cu_ptr + b).to(tl.int32)
        t_total = tl.load(cu_ptr + b + 1).to(tl.int32) - bos
        chunk_off = tl.load(co_ptr + b).to(tl.int32)
        nt = tl.load(co_ptr + b + 1).to(tl.int32) - chunk_off
        sb = 0                    # 变长时输入/h/v_new 的 shape batch 恒 0
    else:
        bos = 0
        t_total = T
        chunk_off = 0
        nt = (T + BT - 1) // BT
        sb = b
    # 注意：输入/h/v_new 一律用 sb 寻址；initial_state/final_state 用 b（= 紧凑 seq 序号）

    kk = tl.arange(0, K)
    tt = tl.arange(0, BT)
    vv = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    h_off = kk[:, None] * V + vv[None, :]

    # h 条带初始化（fp32 常驻）
    if USE_INITIAL:
        h = tl.load(is_ptr + b * sib + hv * sih + h_off).to(tl.float32)
    else:
        h = tl.zeros((K, BLOCK_V), dtype=tl.float32)

    out_dtype = h_ptr.dtype.element_ty

    for i in range(0, nt):
        t0 = bos + i * BT
        actual = tl.minimum(BT, t_total - i * BT)
        tmask = tt < actual

        # 存 chunk i 的起始状态 h[i]（RTNE 回 input dtype）
        tl.store(h_ptr + sb * shb + hv * shh + (chunk_off + i) * shc
                 + kk[:, None] * shk + vv[None, :],
                 h.to(out_dtype))

        # C1+V1: v_new = u - w @ h
        w_t = tl.load(w_ptr + sb * swb + hv * swh + (t0 + tt)[:, None] * swt + kk[None, :],
                      mask=tmask[:, None], other=0.0)
        u_t = tl.load(u_ptr + sb * sub + hv * suh + (t0 + tt)[:, None] * sut + vv[None, :],
                      mask=tmask[:, None], other=0.0)
        v_new = u_t.to(tl.float32) - tl.dot(w_t, h.to(out_dtype))

        tl.store(v_new_ptr + sb * svb + hv * svh + (t0 + tt)[:, None] * svt + vv[None, :],
                 v_new.to(out_dtype), mask=tmask[:, None])

        # V1 衰减 + V2 h 更新
        if USE_G:
            g_t = tl.load(g_ptr + sb * sgb + hv * sgh + t0 + tt,
                          mask=tmask, other=0.0).to(tl.float32)
            g_last = tl.load(g_ptr + sb * sgb + hv * sgh + t0 + actual - 1).to(tl.float32)
            decay = tl.exp(g_last - g_t)                      # [BT]
            v_up = v_new * decay[:, None]
            h = h * tl.exp(g_last)
        else:
            v_up = v_new

        if USE_GK:
            gk_last = tl.load(gk_ptr + sb * sgkb + hv * sgkh
                              + (t0 + actual - 1) * sgkt + kk).to(tl.float32)  # [K]
            h = h * tl.exp(gk_last * LN2)[:, None]
            # kg = k（KDA 的逐 key 预衰减由调用方完成，kernel 不再乘）

        # C2: h += k^T @ v_up（k 转置寻址 load [K, BT]）
        k_t = tl.load(k_ptr + sb * skb + hk * skh + (t0 + tt)[None, :] * skt + kk[:, None],
                      mask=tmask[None, :], other=0.0)
        h += tl.dot(k_t, v_up.to(out_dtype))

    if STORE_FINAL:
        tl.store(fs_ptr + b * sfb + hv * sfh + h_off, h.to(fs_ptr.dtype.element_ty))


def chunk_gated_delta_rule_fwd_h(k, w, u, g=None, gk=None, initial_state=None,
                                 output_final_state=False, chunk_size=64,
                                 cu_seqlens=None, chunk_indices=None,
                                 state_v_first=False, BLOCK_V=128):
    """wrapper：折叠 B 维寻址为 strides；gk/g 至少给一个（对齐 tiling.cpp:91-93）。"""
    assert (g is not None) or (gk is not None), "Either g or gk must be provided."
    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    assert HV % HK == 0
    G = HV // HK
    BT = chunk_size
    BLOCK_V = min(BLOCK_V, V)
    NVB = V // BLOCK_V

    if cu_seqlens is None:
        N, NT_flat = B, (T + BT - 1) // BT
        cu = co = k  # 占位指针，VARLEN=False 不会读
        h = torch.empty((B, HV, NT_flat, K, V), dtype=k.dtype, device=k.device)
    else:
        cu = cu_seqlens.to(torch.int64)
        lens = cu[1:] - cu[:-1]
        nchunks = (lens + BT - 1) // BT
        co = torch.cat([torch.zeros(1, dtype=torch.int64, device=cu.device),
                        nchunks.cumsum(0)])
        N, NT_flat = lens.numel(), int(co[-1])
        h = torch.empty((1, HV, NT_flat, K, V), dtype=k.dtype, device=k.device)

    v_new = torch.empty_like(u)
    fs = None
    if output_final_state or True:  # 骨架恒分配，便于统一 STORE_FINAL 两态
        fs_dtype = initial_state.dtype if initial_state is not None else torch.float32
        fs = torch.empty((N, HV, K, V), dtype=fs_dtype, device=k.device)

    grid = (N * HV * NVB,)
    _chunk_gdn_fwd_h_kernel[grid](
        k, w, u,
        g if g is not None else k,
        gk if gk is not None else k,
        initial_state if initial_state is not None else k,
        h, v_new, fs,
        cu, co,
        HV, G, T,
        k.stride(0), k.stride(1), k.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        g.stride(0) if g is not None else 0, g.stride(1) if g is not None else 0,
        gk.stride(0) if gk is not None else 0, gk.stride(1) if gk is not None else 0,
        gk.stride(2) if gk is not None else 0,
        initial_state.stride(0) if initial_state is not None else 0,
        initial_state.stride(1) if initial_state is not None else 0,
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        fs.stride(0), fs.stride(1),
        K=K, V=V, BT=BT,
        USE_G=g is not None, USE_GK=gk is not None,
        USE_INITIAL=initial_state is not None, STORE_FINAL=True,
        VARLEN=cu_seqlens is not None,
        BLOCK_V=BLOCK_V,
    )
    if state_v_first:
        h = h.transpose(-1, -2)
        fs = fs.transpose(-1, -2)  # 零拷贝 stride 交换（aclnn 的 TransposeLastTwo 在 Triton 不需要）
    return h, v_new, (fs if output_final_state else None)


def ref_chunk_gated_delta_rule_fwd_h(k, w, u, g=None, gk=None, initial_state=None,
                                     chunk_size=64, cu_seqlens=None):
    """golden：复刻 tests/pta/test_fwd_h.py forward_h_trans_cpu，全 fp32 域。
    填充纪律：g 必须 chunk 内 cumsum（单调负衰减），见 mapping.md §7。"""
    dtype_ = k.dtype
    k, w, u = k.float(), w.float(), u.float()
    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    BT = chunk_size
    G = HV // HK

    if cu_seqlens is None:
        lens = [T] * B
    else:
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    N = len(lens)
    NTs = [(l + BT - 1) // BT for l in lens]
    co = [0]
    for c in NTs:
        co.append(co[-1] + c)
    NT_flat = co[-1]

    S = torch.zeros((B if cu_seqlens is None else 1, HV, NT_flat, K, V), dtype=torch.float32)
    v_new_out = torch.zeros_like(u)
    fs = torch.zeros((N, HV, K, V), dtype=torch.float32)

    for n in range(N):
        bos = 0 if cu_seqlens is None else int(cu_seqlens[n])
        sb = n if cu_seqlens is None else 0
        for hv in range(HV):
            h_state = (initial_state[n, hv].float()
                       if initial_state is not None else torch.zeros(K, V))
            for i in range(NTs[n]):
                t0 = bos + i * BT
                actual = min(BT, lens[n] - i * BT)
                S[sb, hv, co[n] + i] = h_state
                kk_ = k[sb, hv // G, t0:t0 + actual]          # [a, K]
                ww = w[sb, hv, t0:t0 + actual]                # [a, K]
                uu = u[sb, hv, t0:t0 + actual]                # [a, V]
                v_new = uu - ww @ h_state
                v_new_out[sb, hv, t0:t0 + actual] = v_new
                v_up = v_new
                if g is not None:
                    g_seg = g[sb, hv, t0:t0 + actual].float()
                    g_last = g_seg[-1]
                    v_up = v_new * torch.exp(g_last - g_seg)[:, None]
                    h_state = h_state * torch.exp(g_last)
                if gk is not None:
                    gk_last = gk[sb, hv, t0 + actual - 1].float()   # [K]
                    h_state = h_state * torch.exp2(gk_last)[:, None]
                    # kg = k（调用方已预衰减）
                h_state = h_state + kk_.t() @ v_up
            fs[n, hv] = h_state
    return S.to(dtype_), v_new_out.to(dtype_), fs
