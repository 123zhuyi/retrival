@triton.jit
def _k_fuse_tlx(
    gm_ptr,  # [NS, N, M3]  f32   gemm_out_mul
    gs_ptr,  # [NS, N]      f32   gemm_out_sqrsum
    sc_ptr,  # [3]           f32   hc_scale
    ba_ptr,  # [M3]          f32   hc_base
    res_ptr,  # [N, M, H]    bf16  residual
    pp_ptr,  # [N, M]        f32   post_mix  (out)
    cp_ptr,  # [N, M*M]      f32   comb_mix  (out)
    lp_ptr,  # [N, H]        bf16  layer_input (out)
    N,
    H,
    eps_rms: tl.constexpr,
    eps_pre: tl.constexpr,
    eps_sk: tl.constexpr,
    pmul: tl.constexpr,
    M: tl.constexpr,
    M3: tl.constexpr,
    BH: tl.constexpr,
    SR: tl.constexpr,
    NS: tl.constexpr,
    NH: tl.constexpr,
):
    """
    Fused 1D kernel: one block per token.
    Steps: RMS → mix_accum → post_mix → sinkhorn → pre_mix → weighted_sum
    The weighted_sum loop uses tlx.async_load for 2-stage software pipelining.
    """
    pid = tl.program_id(0)
    mo = tl.arange(0, M)  # [M]
    ri = mo[:, None]  # [M, 1]
    ci = mo[None, :]  # [1, M]
    ho = tl.arange(0, BH)  # [BH]
    sq = 0.0
    for s in range(NS):
        sq = sq + tl.load(gs_ptr + s * N + pid)
    inv = tl.math.rsqrt(sq / M / H + eps_rms)
    vp = tl.zeros([M], dtype=tl.float32)
    vpo = tl.zeros([M], dtype=tl.float32)
    vc = tl.zeros([M, M], dtype=tl.float32)
    for s in range(NS):
        b = s * N * M3 + pid * M3
        vp = vp + tl.load(gm_ptr + b + mo)
        vpo = vpo + tl.load(gm_ptr + b + M + mo)
        vc = vc + tl.load(gm_ptr + b + 2 * M + ri * M + ci)
    vp = vp * inv
    vpo = vpo * inv
    vc = vc * inv
    s1 = tl.load(sc_ptr + 1)
    pob = tl.load(ba_ptr + M + mo)
    tl.store(pp_ptr + pid * M + mo, tl.sigmoid(vpo * s1 + pob) * pmul)
    s2 = tl.load(sc_ptr + 2)
    cb = tl.load(ba_ptr + 2 * M + ri * M + ci)
    cm = vc * s2 + cb
    mx = tl.max(cm, axis=1)
    cm = tl.exp(cm - mx[:, None])
    rs = tl.sum(cm, axis=1)
    cm = cm / rs[:, None] + eps_sk
    cs = tl.sum(cm, axis=0)
    cm = cm / (cs[None, :] + eps_sk)
    for _ in range(SR - 1):
        rs = tl.sum(cm, axis=1)
        cm = cm / (rs[:, None] + eps_sk)
        cs = tl.sum(cm, axis=0)
        cm = cm / (cs[None, :] + eps_sk)
    tl.store(cp_ptr + pid * M * M + ri * M + ci, cm)
    s0 = tl.load(sc_ptr + 0)
    pb = tl.load(ba_ptr + mo)
    pw = tl.sigmoid(vp * s0 + pb) + eps_pre  # [M]
    rbase = res_ptr + pid * M * H  # Base pointer for this token
    # Build 2D offset pattern: [M, BH]
    # offset[m, bh] = m*H + ih*BH + bh
    off_m = mo[:, None] * H  # [M, 1]
    off_h = ho[None, :]  # [1, BH]
    for ih in range(NH):
        # Load tile: [M, BH] bf16
        tile_ptr = rbase + off_m + ih * BH + off_h
        tile = tl.load(tile_ptr).to(tl.float32)  # [M, BH]
        acc = tl.sum(pw[:, None] * tile, axis=0)  # [BH] f32
        # Store output slice
        tl.store(
            lp_ptr + pid * H + ih * BH + ho,
            acc.to(tl.bfloat16),
        )