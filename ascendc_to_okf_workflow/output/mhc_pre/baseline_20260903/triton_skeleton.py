"""
mhc_pre Triton 骨架（P2 产出，pseudocode 级，可直接翻译为 @triton.jit）

对应 Ascend C：src/mhc/mhc_pre/op_kernel/arch35/{mhc_pre_split_bs,mhc_pre_split_nd,
mhc_pre_m_k_split,mhc_pre_cube_compute,mhc_pre_vector_compute}.h

语义（README L22-46，与 vector_compute.h 逐步核对）：
  invRms = 1/sqrt(mean(x_flat^2) + norm_eps)          # x_flat: [T, K], K = n*D
  xGamma = x_flat * gamma_flat                        # gamma 可选，[n, D] 展平为 [K]
  hMix   = xGamma @ phi^T                             # [T, MAT_N], MAT_N = 2n 或 n^2+2n
  w      = hMix * invRms[:, None]
  段布局（vector_compute.h:107-125）：[pPre(n) | pPost(n) | pRes(n^2)]
  p      = w * alpha_vec + bias_vec                   # alpha/bias 预展开为长度 MAT_N
  hPre   = sigmoid(p_pre) + hc_eps                    # 恒加 hc_eps（vector_compute.h:602）
  hPost  = 2*sigmoid(p_post) (+ hc_eps 仅当无 res, vector_compute.h:641-643)
  hRes   = p_res                                      # 无非线性
  hIn_d  = sum_i hPre_i * x_{i,d}                     # fp32 累加，CAST_RINT 回 b16

布局决策：
1. Ascend C 三 tiling 模式（BS/ND/MK）合并为单 kernel：M 轴 BLOCK_M 分块，
   K 轴 program 内顺序循环（无 split-K / 无跨核同步 / 无 workspace），
   等价于 deterministic 语义（单program累加顺序固定）。
2. phi 按行拆三段（pre: 0..n, post: n..2n, res: 2n..2n+n^2），同一 x tile 做
   三次 tl.dot，分别累加进三个 acc——避免在大 tile 上做列区间重映射。
   三个 acc 的 N 维 padding：BLOCK_NN=16（n≤8）、BLOCK_RES=64（n²≤64）。
"""
import torch
import triton
import triton.language as tl

try:
    import triton.runtime.driver as driver
    NUM_CORE = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"]
except Exception:
    NUM_CORE = 48  # 按目标平台物理核数填写


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128, "BLOCK_D": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_D": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_D": 128}),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_D": 64}),
    ],
    key=["T", "K", "N"],
)
@triton.jit
def mhc_pre_kernel(
    x_ptr, phi_ptr, gamma_ptr, alpha_ptr, bias_ptr,
    hin_ptr, hpost_ptr, hres_ptr, invrms_ptr, hmix_ptr, hpre_ptr,
    T, K, D, MAT_N,
    norm_eps, hc_eps,
    N: tl.constexpr,               # n ∈ {4, 6, 8}
    HAS_GAMMA: tl.constexpr,
    HAS_RESI: tl.constexpr,        # alpha.shape == [3]
    OUT_FLAG: tl.constexpr,
    DOT_PRECISION: tl.constexpr,   # "ieee"(impl_mode=0) / "hf32"(impl_mode=1)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_NN: tl.constexpr,        # = 16（n ≤ 8 padding 到 dot 最小维）
    BLOCK_RES: tl.constexpr,       # = 64（n² ≤ 64 padding）
):
    NUM_BLOCKS = tl.cdiv(T, BLOCK_M)
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    offs_nn = tl.arange(0, BLOCK_NN)
    nn_valid = offs_nn < N                       # pre/post 段列 mask
    offs_res = tl.arange(0, BLOCK_RES)
    res_valid = offs_res < N * N                 # res 段列 mask

    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        offs_m = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        m_valid = offs_m < T

        # ===== 阶段 1：K 循环 { cast fp32 + gamma + Σx² + 三次 dot } =====
        # 对应 Ascend C：V0Prologue(cast+γ+Σx²) + Mmad 累加 + partial 归并
        acc_pre = tl.zeros([BLOCK_M, BLOCK_NN], dtype=tl.float32)
        acc_post = tl.zeros([BLOCK_M, BLOCK_NN], dtype=tl.float32)
        acc_res = tl.zeros([BLOCK_M, BLOCK_RES], dtype=tl.float32)
        sumsq = tl.zeros([BLOCK_M], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_valid = offs_k < K
            mask_mk = m_valid[:, None] & k_valid[None, :]
            # x tile [BLOCK_M, BLOCK_K]，对应 DataCopyPad + Reg::Cast B16→FP32
            x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                        mask=mask_mk, other=0.0).to(tl.float32)
            if HAS_GAMMA:
                g = tl.load(gamma_ptr + offs_k, mask=k_valid, other=0.0)  # gamma 展平 [K]
                xg = x * g[None, :]
            else:
                xg = x
            sumsq += tl.sum(x * x, axis=1)          # Σx²（vector_compute.h:406-408）
            # phi 是 [MAT_N, K] 行主序；dot 需要 [BLOCK_K, *] → 转置读（编译器处理 Nd2Nz 等价布局）
            phi_pre = tl.load(phi_ptr + offs_nn[None, :] * K + offs_k[:, None],
                              mask=k_valid[:, None] & nn_valid[None, :], other=0.0)
            phi_post = tl.load(phi_ptr + (N + offs_nn[None, :]) * K + offs_k[:, None],
                               mask=k_valid[:, None] & nn_valid[None, :], other=0.0)
            acc_pre += tl.dot(xg, phi_pre, input_precision=DOT_PRECISION)
            acc_post += tl.dot(xg, phi_post, input_precision=DOT_PRECISION)
            if HAS_RESI:
                phi_res = tl.load(phi_ptr + (2 * N + offs_res[None, :]) * K + offs_k[:, None],
                                  mask=k_valid[:, None] & res_valid[None, :], other=0.0)
                acc_res += tl.dot(xg, phi_res, input_precision=DOT_PRECISION)

        # invRms = 1/sqrt(mean + eps)（vector_compute.h:429-433：Muls(1/K)→Adds(eps)→Sqrt→Div）
        inv_rms = 1.0 / tl.sqrt(sumsq / K + norm_eps)

        # ===== 阶段 2：w = hMix * invRms；分段 ×alpha +bias；门控 =====
        # 对应 MhcPreVFPostSegments（vector_compute.h:103-127）+ HPre/HPost（L582-656）
        # alpha_vec/bias_vec 已由 wrapper 预展开：[a0]*n+[a1]*n(+[a2]*n²)，bias 原长度 MAT_N
        a_pre = tl.load(alpha_ptr + offs_nn, mask=nn_valid, other=0.0)
        b_pre = tl.load(bias_ptr + offs_nn, mask=nn_valid, other=0.0)
        a_post = tl.load(alpha_ptr + N + offs_nn, mask=nn_valid, other=0.0)
        b_post = tl.load(bias_ptr + N + offs_nn, mask=nn_valid, other=0.0)

        p_pre = acc_pre * inv_rms[:, None] * a_pre[None, :] + b_pre[None, :]
        p_post = acc_post * inv_rms[:, None] * a_post[None, :] + b_post[None, :]
        hpre = 1.0 / (1.0 + tl.exp(-p_pre)) + hc_eps            # σ+hc_eps（恒加）
        hpost = 2.0 / (1.0 + tl.exp(-p_post))                   # 2σ
        if not HAS_RESI:
            hpost = hpost + hc_eps                              # 无 res 才加 hc_eps

        mask_m_nn = m_valid[:, None] & nn_valid[None, :]
        tl.store(hpost_ptr + offs_m[:, None] * N + offs_nn[None, :], hpost, mask=mask_m_nn)
        if OUT_FLAG:
            tl.store(invrms_ptr + offs_m, inv_rms, mask=m_valid)
            tl.store(hpre_ptr + offs_m[:, None] * N + offs_nn[None, :], hpre, mask=mask_m_nn)
            # hMix = acc（未乘 invRms 的 matmul 原值），按 [pre|post|res] 段布局写回
            tl.store(hmix_ptr + offs_m[:, None] * MAT_N + offs_nn[None, :],
                     acc_pre, mask=mask_m_nn)
            tl.store(hmix_ptr + offs_m[:, None] * MAT_N + N + offs_nn[None, :],
                     acc_post, mask=mask_m_nn)
        if HAS_RESI:
            a_res = tl.load(alpha_ptr + 2 * N + offs_res, mask=res_valid, other=0.0)
            b_res = tl.load(bias_ptr + 2 * N + offs_res, mask=res_valid, other=0.0)
            p_res = acc_res * inv_rms[:, None] * a_res[None, :] + b_res[None, :]
            mask_m_res = m_valid[:, None] & res_valid[None, :]
            # hRes shape (T, n, n)，行内 n*n 连续
            tl.store(hres_ptr + offs_m[:, None] * (N * N) + offs_res[None, :],
                     p_res, mask=mask_m_res)
            if OUT_FLAG:
                tl.store(hmix_ptr + offs_m[:, None] * MAT_N + 2 * N + offs_res[None, :],
                         acc_res, mask=mask_m_res)

        # ===== 阶段 3：hIn_d = Σ_i hPre_i · x_{i,d}（VFDoV1ProcessHInRows, L658-726） =====
        # hpre tile [BLOCK_M, BLOCK_NN]，列 i<n 有效；x 视角 [T, n, D]
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            d_valid = offs_d < D
            mask_md = m_valid[:, None] & d_valid[None, :]
            hin_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
            for i in tl.static_range(BLOCK_NN):     # n ≤ 8 全展开（对应 HEAD_NUM 模板展开）
                xi = tl.load(x_ptr + offs_m[:, None] * (N * D) + i * D + offs_d[None, :],
                             mask=mask_md, other=0.0).to(tl.float32)
                # 取 hpre 第 i 列并广播（i ≥ N 时置零）
                hi = tl.sum(tl.where(offs_nn[None, :] == i, hpre, 0.0), axis=1)
                hi = tl.where(i < N, hi, 0.0)
                hin_acc += hi[:, None] * xi         # fp32 累加（vector_compute.h:699-719）
            tl.store(hin_ptr + offs_m[:, None] * D + offs_d[None, :],
                     hin_acc.to(hin_ptr.dtype.element_ty),   # RTNE ≈ CAST_RINT
                     mask=mask_md)


def mhc_pre(x, phi, alpha, bias, gamma=None, norm_eps=1e-6, hc_eps=1e-6,
            out_flag=False, impl_mode=0):
    """
    x: (B,S,n,D) 或 (T,n,D)，bf16/fp16；phi: (MAT_N, nD) fp32；
    alpha: (3)/(2) fp32；bias: (MAT_N,) fp32；gamma: (n,D) fp32 可选。
    返回 (h_in, h_post, h_res[, inv_rms, h_mix, h_pre])。
    """
    assert x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16)
    shape = x.shape
    n, D = shape[-2], shape[-1]
    assert n in (4, 6, 8) and 16 <= D <= 16384 and D % 16 == 0
    T = shape[0] if len(shape) == 3 else shape[0] * shape[1]
    K = n * D
    has_resi = alpha.numel() == 3
    mat_n = n * n + 2 * n if has_resi else 2 * n
    assert phi.shape == (mat_n, K) and bias.numel() == mat_n

    # alpha 预展开：(2|3) → [a0]*n + [a1]*n (+ [a2]*n²)，对应 VFExpandAlpha
    a = alpha.float().cpu()
    alpha_vec = torch.cat([a[0].repeat(n), a[1].repeat(n),
                           a[2].repeat(n * n) if has_resi else a.new_empty(0)]).to(x.device)

    hin = torch.empty(shape[:-2] + (D,), dtype=x.dtype, device=x.device)
    hpost = torch.empty(shape[:-2] + (n,), dtype=torch.float32, device=x.device)
    hres = (torch.empty(shape[:-2] + (n, n), dtype=torch.float32, device=x.device)
            if has_resi else torch.empty(0, dtype=torch.float32, device=x.device))
    if out_flag:
        inv_rms = torch.empty(shape[:-2], dtype=torch.float32, device=x.device)
        h_mix = torch.empty(shape[:-2] + (mat_n,), dtype=torch.float32, device=x.device)
        h_pre = torch.empty(shape[:-2] + (n,), dtype=torch.float32, device=x.device)
    else:
        inv_rms = h_mix = h_pre = torch.empty(1, dtype=torch.float32, device=x.device)

    mhc_pre_kernel[(NUM_CORE,)](
        x, phi, gamma if gamma is not None else x, alpha_vec, bias.float(),
        hin, hpost, hres, inv_rms, h_mix, h_pre,
        T, K, D, mat_n, norm_eps, hc_eps,
        N=n, HAS_GAMMA=gamma is not None, HAS_RESI=has_resi,
        OUT_FLAG=out_flag, DOT_PRECISION="hf32" if impl_mode == 1 else "ieee",
        BLOCK_NN=16, BLOCK_RES=64,
    )
    return hin, hpost, hres, inv_rms, h_mix, h_pre
