"""P2 Triton 骨架 — mhc_pre（新流程版，2026-09-14）

语义（README L22-46，与 vector_compute.h 逐步核对，2026-09-14 复验 5/5 锚点）：
  invRms = 1/sqrt(mean(x_flat^2) + norm_eps)          # x_flat: [T, K], K = n*D
  xGamma = x_flat * gamma_flat                        # gamma 可选，[n, D] 展平为 [K]
  hMix   = xGamma @ phi^T                             # [T, MAT_N], MAT_N = 2n 或 n^2+2n
  w      = hMix * invRms[:, None]
  段布局（vector_compute.h:100-128）：[pPre(n) | pPost(n) | pRes(n^2)]
  p      = w * alpha_vec + bias_vec                   # alpha/bias 预展开为长度 MAT_N
  hPre   = sigmoid(p_pre) + hc_eps                    # 恒加（vector_compute.h:602）
  hPost  = 2*sigmoid(p_post) (+ hc_eps 仅当无 res, vector_compute.h:640-643)
  hRes   = p_res                                      # 无非线性
  hIn_d  = sum_i hPre_i * x_{i,d}                     # fp32 累加，CAST_RINT 回 b16

设计决策与 backend_facts 取证对照（mapping.md §0.5 检索记录）：
- 三 tiling 模式合并为单 kernel：K 轴 program 内顺序循环，消解 split-K/atomic/跨核
  flag/workspace/两阶段（语义等价替代，天然确定）——结构性论证，非 C-012 等价断言
- grid = num_aicore（含 tl.dot = CV 融合算子，C-011 / operator_class_patterns [DOC]；
  旧版取 num_vectorcore 已修正）。核数运行时查询、禁止写死（hardware_platforms §3：
  950 AIC 4~36 跨度 9 倍）
- **int64 安全寻址**（mapping §0.6-3，旧版潜伏 bug）：T·K 最大 4.3e9 > int32，
  13 个 benchmark case 中 6 个溢出。块基址用 int64 标量（不占向量 lane，不受 i64
  标量化影响 [SRC dtype_support §2.1]），块内偏移 int32（BLOCK_M·K ≤ 1.7e7 安全）
- bf16 load 后立即 cast fp32：950 regbase 向量指令无 bf16
  [SRC RegbaseUtils.cpp isHIVMSupportedType]，与源码 Reg::Cast B16_TO_FP32 一致
- mask 保持 int32 比较（C-002 fp32 导出 mask 非确定 store 污染）；hIn 取列用 i32 EQ
  比较——向量安全（仅 i32 LT/GT/LE/GE 标量化 [SRC VCmpOp impl]）
- 无 num_stages/num_warps（C-006 运行时不生效）；双缓冲/预取由 multibuffer 承接
  （bishengir 默认 true [SRC Options.td EnableAutoMultiBuffer]）；CV 同步由
  sync_solver（默认 true [SRC]）承接，不手写 al.sync_block_*
- autotune：本 kernel 非 µs 级（K 至 131072 重 matmul），C-007 不适用故保留；
  守 C-014 三约束（grid 为常量不需 lambda）；config 内不得出现 num_stages/num_warps
- UB 预算按 950 = 248KB 可用校核 [SRC hardware_platforms]（旧版 192KB 是 A2 值）：
  n≤8 峰值 ~110KB 富余 2.3 倍；n=32 时 BLOCK_K≤32 且 BLOCK_M≤16（见 mapping §4）
- input_precision="hf32"（impl_mode=1）[UNVERIFIED]，默认 "ieee" 对齐 FP32 标杆

preflight（backend_contracts 8 项 + dtype_support 4a~4d，2026-09-14 对本文件执行）：
12 项 0 真命中（头注禁令文字已剥离后扫描；wrapper host-Python 不违禁）。
"""
import torch
import triton
import triton.language as tl


def _aicore_num() -> int:
    """运行时查询 AIC（Cube）核数；CV 融合算子 grid 取 num_aicore（C-011）。
    禁止写死（hardware_platforms §3：950 AIC 4~36 跨度 9 倍）；
    fallback 24 仅为兜底，查询失败时 grid 可能错配——P4 须确认查询成功。"""
    try:
        import triton.runtime.driver as driver
        return driver.active.utils.get_device_properties(
            torch.npu.current_device())["num_aicore"]
    except Exception:
        return 24


@triton.autotune(
    configs=[
        # 不得引入 num_stages/num_warps（C-006 运行时不生效）
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128, "BLOCK_D": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_D": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_D": 128}),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_D": 64}),
        # n=32（BLOCK_RES=1024）时 UB 临界：BLOCK_K≤32 且 BLOCK_M≤16（mapping §4）
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32, "BLOCK_D": 64}),
    ],
    key=["T", "K", "N"],
)
@triton.jit
def mhc_pre_kernel(
    x_ptr, phi_ptr, gamma_ptr, alpha_ptr, bias_ptr,
    hin_ptr, hpost_ptr, hres_ptr, invrms_ptr, hmix_ptr, hpre_ptr,
    T, K, D, MAT_N,
    norm_eps, hc_eps,
    N: tl.constexpr,               # n，benchmark {4,8,12,16,24,32}（不硬编码 Ascend C n≤8）
    HAS_GAMMA: tl.constexpr,
    HAS_RESI: tl.constexpr,        # alpha.shape == [3]
    OUT_FLAG: tl.constexpr,
    DOT_PRECISION: tl.constexpr,   # "ieee"(impl_mode=0) / "hf32"(impl_mode=1, UNVERIFIED)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_NN: tl.constexpr,        # max(16, next_pow2(n))，dot 最小维 16
    BLOCK_RES: tl.constexpr,       # max(16, next_pow2(n²))
):
    NUM_BLOCKS = tl.cdiv(T, BLOCK_M)
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    offs_nn = tl.arange(0, BLOCK_NN)
    nn_valid = offs_nn < N                       # pre/post 段列 mask（int32 比较，C-002）
    offs_res = tl.arange(0, BLOCK_RES)
    res_valid = offs_res < N * N                 # res 段列 mask

    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        offs_m = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        m_valid = offs_m < T

        # int64 安全寻址：块基址为 int64 标量（T·K 可达 4.3e9 > int32）；
        # 块内偏移 int32（BLOCK_M·K ≤ 1.7e7 安全）——mapping §0.6-3
        x_base = blk.to(tl.int64) * BLOCK_M * K

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
            # x tile [BLOCK_M, BLOCK_K]：DataCopyPad + Reg::Cast B16→FP32
            # （950 无 bf16 向量指令，立即 cast fp32 是必须 [SRC]）
            x_off = tl.arange(0, BLOCK_M)[:, None] * K + offs_k[None, :]
            x = tl.load(x_ptr + x_base + x_off,
                        mask=mask_mk, other=0.0).to(tl.float32)
            if HAS_GAMMA:
                g = tl.load(gamma_ptr + offs_k, mask=k_valid, other=0.0)  # 展平 [K]
                xg = x * g[None, :]
            else:
                xg = x
            sumsq += tl.sum(x * x, axis=1)          # Σx²（fp32 vreduce 向量安全 [SRC]）
            # phi [MAT_N, K] 行主序；dot 需要 [BLOCK_K, *] → 转置读（编译器处理 Nd2Nz 等价）
            # phi 偏移最大 MAT_N·K = 1088×131072 = 1.4e8 < int32，安全
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

        # invRms：Muls(scaleMean)→Adds(normEps)→Sqrt→Div（vector_compute.h:429-433 已复验）
        inv_rms = 1.0 / tl.sqrt(sumsq * (1.0 / K) + norm_eps)

        # ===== 阶段 2：w = hMix·invRms；分段 ×alpha+bias；门控 =====
        # 段布局 [pPre(n)|pPost(n)|pRes(n²)]（vector_compute.h:100-128 已复验）
        a_pre = tl.load(alpha_ptr + offs_nn, mask=nn_valid, other=0.0)
        b_pre = tl.load(bias_ptr + offs_nn, mask=nn_valid, other=0.0)
        a_post = tl.load(alpha_ptr + N + offs_nn, mask=nn_valid, other=0.0)
        b_post = tl.load(bias_ptr + N + offs_nn, mask=nn_valid, other=0.0)

        p_pre = acc_pre * inv_rms[:, None] * a_pre[None, :] + b_pre[None, :]
        p_post = acc_post * inv_rms[:, None] * a_post[None, :] + b_post[None, :]
        hpre = 1.0 / (1.0 + tl.exp(-p_pre)) + hc_eps            # σ+hc_eps（恒加，L602）
        hpost = 2.0 / (1.0 + tl.exp(-p_post))                   # 2σ（fp32 VExp/VDiv 原生）
        if not HAS_RESI:
            hpost = hpost + hc_eps                              # 仅无 res 加（L640-643）

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
            tl.store(hres_ptr + offs_m[:, None] * (N * N) + offs_res[None, :],
                     p_res, mask=mask_m_res)
            if OUT_FLAG:
                tl.store(hmix_ptr + offs_m[:, None] * MAT_N + 2 * N + offs_res[None, :],
                         acc_res, mask=mask_m_res)

        # ===== 阶段 3：hIn_d = Σ_i hPre_i·x_{i,d}（fp32 累加，CAST_RINT 回 b16） =====
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            d_valid = offs_d < D
            mask_md = m_valid[:, None] & d_valid[None, :]
            hin_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
            for i in tl.static_range(BLOCK_NN):     # n 头全展开（对应 HEAD_NUM 模板展开）
                # 块内偏移 int32：BLOCK_M·(N·D) ≤ 1.7e7 安全；基址 x_base 复用
                xi_off = tl.arange(0, BLOCK_M)[:, None] * (N * D) + i * D + offs_d[None, :]
                xi = tl.load(x_ptr + x_base + xi_off,
                             mask=mask_md, other=0.0).to(tl.float32)
                # 取 hpre 第 i 列：i32 EQ 比较向量安全（仅 i32 不等值标量化 [SRC VCmpOp]）
                hi = tl.sum(tl.where(offs_nn[None, :] == i, hpre, 0.0), axis=1)
                hi = tl.where(i < N, hi, 0.0)
                hin_acc += hi[:, None] * xi         # fp32 累加（vector_compute.h:699-719）
            # hin 偏移最大 T·D = 262144×4096 = 1.07e9 < int32，安全
            tl.store(hin_ptr + offs_m[:, None] * D + offs_d[None, :],
                     hin_acc.to(hin_ptr.dtype.element_ty),   # RTNE ≈ CAST_RINT
                     mask=mask_md)


def mhc_pre(x, phi, alpha, bias, gamma=None, norm_eps=1e-6, hc_eps=1e-6,
            out_flag=False, impl_mode=0):
    """
    x: (B,S,n,D) 或 (T,n,D)，bf16/fp16；phi: (MAT_N, nD) fp32；
    alpha: (3)/(2) fp32；bias: (MAT_N,) fp32；gamma: (n,D) fp32 可选。
    返回 (h_in, h_post, h_res[, inv_rms, h_mix, h_pre])。
    n 范围以 benchmark 为准（{4,8,12,16,24,32}），不硬编码 Ascend C 的 n∈{4,6,8}。
    """
    assert x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16)
    shape = x.shape
    n, D = shape[-2], shape[-1]
    assert D % 16 == 0
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

    grid = (_aicore_num(),)   # CV 融合算子 grid=num_aicore（C-011）；常量 grid 不需 lambda
    mhc_pre_kernel[grid](
        x, phi, gamma if gamma is not None else x, alpha_vec, bias.float(),
        hin, hpost, hres, inv_rms, h_mix, h_pre,
        T, K, D, mat_n, norm_eps, hc_eps,
        N=n, HAS_GAMMA=gamma is not None, HAS_RESI=has_resi,
        OUT_FLAG=out_flag, DOT_PRECISION="hf32" if impl_mode == 1 else "ieee",
        BLOCK_NN=max(16, triton.next_power_of_2(n)),
        BLOCK_RES=max(16, triton.next_power_of_2(n * n)),
    )
    return hin, hpost, hres, inv_rms, h_mix, h_pre
