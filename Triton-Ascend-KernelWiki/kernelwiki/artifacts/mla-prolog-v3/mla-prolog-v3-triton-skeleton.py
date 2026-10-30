# mla_prolog_v3 - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：DeepSeek V2/V3 MLA 前处理融合算子
#   4 路 matmul（下采样 W^{DQ}、拼接 W^{DKV}+W^{KR}、拼接 W^{UQ}+W^{QR}、上采样 W^{UK}）
#   + 2 个 RmsNorm + 2 个 RoPE + KVCache/KRCache 更新
# CANN 专有机制（Cube-Vector 6 同步点、CrossCoreSetFlag/WaitFlag、FRACTAL_NZ 物理格式、MTE DataCopyPad）
# 不在此体现。
# 来源：attention/mla_prolog_v3/ (commit d28d8f9...), docs/MlaPrologV3算子设计介绍.md.

import triton
import triton.language as tl


@triton.jit
def mla_prolog_v3_kernel(
    # pointers
    token_x_ptr,            # [T, He] 或 [B, S, He]
    weight_dq_ptr,          # [He, Hcq]  W^{DQ}
    weight_uq_qr_ptr,       # [Hcq, Hckv + Dr]  [W^{UQ}|W^{QR}] 拼接
    weight_uk_ptr,          # [Hckv, N * Hckv]  W^{UK}
    weight_dkv_kr_ptr,      # [He, Hckv + Dr]  [W^{DKV}|W^{KR}] 拼接
    rmsnorm_gamma_cq_ptr,
    rmsnorm_gamma_ckv_ptr,
    rope_sin_ptr, rope_cos_ptr,
    kv_cache_ptr,           # KVCache（in/out）
    kr_cache_ptr,           # KRCache（in/out）
    cache_index_ptr,        # [B, S] 或 [T,]
    query_out_ptr,          # q^N [T, N, Hckv] 或 [B, S, N, Hckv]
    query_rope_ptr,         # q^R_rope [T, N, Dr] 或 [B, S, N, Dr]
    # shape / scalars
    He: tl.constexpr,       # token_x hidden dim
    Hcq: tl.constexpr,      # c^Q head dim
    Hckv: tl.constexpr,     # c^{KV} head dim
    Dr: tl.constexpr,       # RoPE head dim
    alpha_q: tl.constexpr,  # qcQrScale
    alpha_kv: tl.constexpr, # kcScale
    rmsnorm_eps_cq: tl.constexpr,
    rmsnorm_eps_ckv: tl.constexpr,
    # feature flags (constexpr)
    CACHE_MODE: tl.constexpr,   # 0:BNSD / 1:PA_BSND / 2:PA_NZ
    QUANT_MODE: tl.constexpr,   # 量化模式
    SPLIT_M: tl.constexpr,      # 0:切N / 1:切M
):
    pid = tl.program_id(0)  # 沿 M 轴（BS 合轴的 token 维）或 N 轴（head 维）切分

    # 1. MatmulCq: x · W^{DQ} -> c^Q  [BS, Hcq]
    #    Triton: tl.dot(x, W_dq, out_dtype=tl.float32)
    x = tl.load(token_x_ptr + pid * He + tl.arange(0, He))  # [He]
    w_dq = tl.load(weight_dq_ptr + tl.arange(0, He)[:, None] * Hcq + tl.arange(0, Hcq)[None, :])  # [He, Hcq]
    c_q = tl.dot(x[None, :], w_dq).to(tl.float32)  # [1, Hcq]

    # 2. RmsNormCq: RmsNorm(c^Q) -> c_norm^Q (设计介绍 L180-196)
    c_q = c_q * alpha_q  # 尺度矫正因子
    c_q_mean = tl.sum(c_q * c_q, axis=1) / Hcq
    c_q_norm = c_q * tl.rsqrt(c_q_mean + rmsnorm_eps_cq)
    gamma_cq = tl.load(rmsnorm_gamma_cq_ptr + tl.arange(0, Hcq))
    c_q_norm = c_q_norm * gamma_cq  # [1, Hcq]

    # 3. MatmulCkvKr: x · [W^{DKV}|W^{KR}] -> [c^{KV}|k^R]  [BS, Hckv + Dr]
    w_dkv_kr = tl.load(weight_dkv_kr_ptr + tl.arange(0, He)[:, None] * (Hckv + Dr) + tl.arange(0, Hckv + Dr)[None, :])
    ckvkr = tl.dot(x[None, :], w_dkv_kr).to(tl.float32)  # [1, Hckv + Dr]
    c_kv, k_r = ckvkr[:, :Hckv], ckvkr[:, Hckv:]  # split

    # 4. RmsNormCkv + RoPE k^R
    c_kv = c_kv * alpha_kv
    ckv_mean = tl.sum(c_kv * c_kv, axis=1) / Hckv
    c_kv_norm = c_kv * tl.rsqrt(ckv_mean + rmsnorm_eps_ckv)
    gamma_ckv = tl.load(rmsnorm_gamma_ckv_ptr + tl.arange(0, Hckv))
    c_kv_norm = c_kv_norm * gamma_ckv  # [1, Hckv]

    # RoPE(k^R) -> k^R_rope (设计介绍 L275-415)
    # 偶奇拆分: k_r_even = k_r[:, 0::2], k_r_odd = k_r[:, 1::2]
    # k_r_rope = [k_r_even * cos - k_r_odd * sin | k_r_odd * cos + k_r_even * sin]
    k_r_rope = _rope(k_r, rope_sin_ptr, rope_cos_ptr, pid, Dr)  # 伪过程

    # KRCache update: kr_cache[cache_index] = k^R_rope
    cache_idx = tl.load(cache_index_ptr + pid)  # cacheIndex 映射
    tl.store(kr_cache_ptr + cache_idx * Dr + tl.arange(0, Dr), k_r_rope.to(kr_cache_ptr.dtype.element_ty()))

    # 5. MatmulQcQr: c_norm^Q · [W^{UQ}|W^{QR}] -> [q^C|q^R]  [BS, Hckv + Dr]
    w_uq_qr = tl.load(weight_uq_qr_ptr + tl.arange(0, Hcq)[:, None] * (Hckv + Dr) + tl.arange(0, Hckv + Dr)[None, :])
    qcqr = tl.dot(c_q_norm, w_uq_qr).to(tl.float32)  # [1, Hckv + Dr]
    q_c, q_r = qcqr[:, :Hckv], qcqr[:, Hckv:]  # split

    # 6. RoPE q^R -> q^R_rope
    q_r_rope = _rope(q_r, rope_sin_ptr, rope_cos_ptr, pid, Dr)

    # 7. MatmulQn: q^C · W^{UK} -> q^N  [BS, N, Hckv]
    w_uk = tl.load(weight_uk_ptr + tl.arange(0, Hckv)[:, None] * (N * Hckv) + tl.arange(0, N * Hckv)[None, :])
    q_n = tl.dot(q_c, w_uk).to(tl.float32)  # [1, N * Hckv]
    q_n = q_n.reshape([1, N, Hckv])

    # 8. KVCache update: kv_cache[cache_index] = c^{KV}_norm
    tl.store(kv_cache_ptr + cache_idx * Hckv + tl.arange(0, Hckv), c_kv_norm.to(kv_cache_ptr.dtype.element_ty()))

    # 输出 query 和 query_rope
    tl.store(query_out_ptr + pid * N * Hckv + tl.arange(0, N * Hckv), q_n.to(query_out_ptr.dtype.element_ty()))
    tl.store(query_rope_ptr + pid * N * Dr + tl.arange(0, N * Dr), q_r_rope.to(query_rope_ptr.dtype.element_ty()))


# RoPE 辅助函数（设计介绍 L361-414）
# 偶奇拆分 -> * cos + 翻转 * sin -> 合并
@triton.jit
def _rope(x, sin_ptr, cos_ptr, token_idx, Dr: tl.constexpr):
    half = Dr // 2
    x_even = x[:, 0::2]  # [1, half]
    x_odd = x[:, 1::2]   # [1, half]
    cos = tl.load(cos_ptr + token_idx * Dr + tl.arange(0, half))
    sin = tl.load(sin_ptr + token_idx * Dr + tl.arange(0, half))
    # 公式(15): [x_even * cos - x_odd * sin | x_odd * cos + x_even * sin]
    res_even = x_even * cos - x_odd * sin
    res_odd = x_odd * cos + x_even * sin
    # 交错合并回 [D] 维
    res = tl.zeros([Dr], dtype=tl.float32)
    res = tl.where(tl.arange(0, Dr) % 2 == 0, res_even, res_odd)
    return res

# grid = (BS,)  # 沿 BS 合轴（M 轴）切分，每个 program 处理一个 token
# 量化分支（QUANT_MODE）：
#   - 动态量化：tl.max + scale = max/127 + round(x/scale)
#   - 静态量化 per-channel：tl.dot 后 *= scale
# 空 tensor 分支（EMPTY_TENSOR_MODE）：
#   - 1: KVCache+KRCache 空 -> 不更新 cache
#   - 2: Query 空 -> 不输出 query/query_rope
# actualSeqLen 分支：使能时 per-batch offset + mask