# mla_prolog - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：MLA 前处理 prolog（4 matmul + RmsNorm + RoPE + Cache scatter）。
# 与 mla_preprocess 不同：本算子为 V1，权重 W^DQ/W^UQ/W^QR/W^DKV/W^KR 独立（非 W^DQKV 合并），
#   且支持 950(FP8) + A2/A3；mla_preprocess 仅 A2/A3，权重更聚合。
# CANN 专有机制（6 CV sync event/ND↔NZ/MIX_AIC_1_2/FRACTAL_NZ）不在此体现。
# 来源：attention/mla_prolog/ (commit d28d8f9...), docs/MlaProlog算子设计介绍.md L9-146.

import triton
import triton.language as tl


@triton.jit
def mla_prolog_kernel(
    # pointers
    x_ptr,              # [T, He] 或 [B, S, He] BF16/INT8
    w_dq_ptr,           # [He, Hcq] NZ
    w_uq_qr_ptr,        # [Hcq, N*(D+Dr)] NZ
    w_uk_ptr,           # [N, D, Hckv] ND
    w_dkv_kr_ptr,       # [He, Hckv+Dr] NZ
    gamma_cq, gamma_ckv,
    cos_ptr, sin_ptr,   # [T, Dr] 或 [B, S, Dr]
    cache_index_ptr,    # [T] 或 [B, S] INT64
    kv_cache_ptr, kr_cache_ptr,
    query_out_ptr, query_rope_out_ptr, kv_cache_out_ptr, kr_cache_out_ptr,
    # scalars
    token_num, hidden_size, eps,
    head_num, head_dim, head_dim_rope,
    # meta (constexpr) -- 值 source-reported
    BLOCK_T: tl.constexpr,    # M 轴（token 块）
    BLOCK_M: tl.constexpr,    # matmul 基本块 M
    BLOCK_N: tl.constexpr,    # matmul 基本块 N
    BLOCK_K: tl.constexpr,    # matmul 基本块 K
    HCQ: tl.constexpr,        # 压缩 Q 维度
    HCKV: tl.constexpr,       # 压缩 KV 维度
    DR: tl.constexpr,         # rope 维度
    D: tl.constexpr,          # nope 维度
    N: tl.constexpr,          # head num
    SCENARIO: tl.constexpr,   # 0=NO_QUANT, 1=QUANT
    QUANT_MODE: tl.constexpr, # 0=KV_NO_QUANT, 1=KV_QUANT
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < token_num
    offs_he = tl.arange(0, hidden_size)

    # ---- 1. MatmulCq: x·W^DQ -> [T, Hcq] ----
    x = tl.load(x_ptr + offs_t[:, None] * hidden_size + offs_he[None, :],
                mask=t_mask[:, None], other=0.0).to(tl.float32)
    # W^DQ: [He, Hcq] 转置 -> [Hcq, He]
    w_dq = tl.load(w_dq_ptr + offs_he[:, None] * HCQ + tl.arange(0, HCQ)[None, :])
    cq = tl.dot(x, w_dq.to(tl.float32))  # [T, Hcq]

    # ---- 2. RmsNormCq ----
    cq_norm = rmsnorm(cq, gamma_cq, eps, HCQ)  # 伪过程

    # ---- 3. MatmulCkvKr: x·[W^DKV|W^KR] -> [T, Hckv+Dr] ----
    offs_ckv_kr = tl.arange(0, HCKV + DR)
    w_dkv_kr = tl.load(w_dkv_kr_ptr + offs_he[:, None] * (HCKV+DR) + offs_ckv_kr[None, :])
    ckv_kr = tl.dot(x, w_dkv_kr.to(tl.float32))  # [T, Hckv+Dr]
    # split -> ckv [T, Hckv] + kr [T, Dr]
    ckv, kr = ckv_kr_split(ckv_kr, HCKV, DR)

    # ---- 4. RmsNormCkv ----
    ckv_norm = rmsnorm(ckv, gamma_ckv, eps, HCKV)  # 伪过程

    # ---- 5. Cache scatter: k^C ----
    idx = tl.load(cache_index_ptr + offs_t, mask=t_mask, other=0)
    tl.store(kv_cache_out_ptr + idx[:, None] * HCKV + tl.arange(0, HCKV)[None, :],
             ckv_norm, mask=t_mask[:, None])

    # ---- 6. MatmulQcQr: cq_norm·[W^UQ|W^QR] -> [T, N*(D+Dr)] ----
    w_uq_qr = tl.load(w_uq_qr_ptr + tl.arange(0, HCQ)[:, None] * N*(D+DR) + tl.arange(0, N*(D+DR))[None, :])
    qc_qr = tl.dot(cq_norm, w_uq_qr.to(tl.float32))
    # split -> qc [T, N*D] + qr [T, N*Dr]
    qc, qr = qc_qr_split(qc_qr, N, D, DR)

    # ---- 7. MatmulQn: qc·W^UK -> [T, N*Hckv] ----
    # W^UK: [N, D, Hckv] -> 展平为 [N*D, Hckv]
    w_uk = tl.load(w_uk_ptr + tl.arange(0, N*D)[:, None] * HCKV + tl.arange(0, HCKV)[None, :])
    qn = tl.dot(qc, w_uk.to(tl.float32))  # q^N -> queryOut

    # ---- 8. RoPE Qr ----
    qr_rope = rope(qr, cos_ptr, sin_ptr, offs_t, DR)  # 伪过程 -> q^R

    # ---- 9. RoPE Kr + cache scatter ----
    kr_rope = rope(kr, cos_ptr, sin_ptr, offs_t, DR)
    tl.store(kr_cache_out_ptr + idx[:, None] * DR + tl.arange(0, DR)[None, :],
             kr_rope, mask=t_mask[:, None])

    # 输出
    tl.store(query_out_ptr + offs_t[:, None] * N*HCKV + tl.arange(0, N*HCKV)[None, :],
             qn, mask=t_mask[:, None])
    tl.store(query_rope_out_ptr + offs_t[:, None] * N*DR + tl.arange(0, N*DR)[None, :],
             qr_rope, mask=t_mask[:, None])


# 辅助（伪过程）：
#   rmsnorm(x, gamma, eps, dim) = x / sqrt(mean(x^2) + eps) * gamma
#   rope(x, cos, sin, offs, dim) = x * cos + rotate_half(x) * sin
#   ckv_kr_split / qc_qr_split 按列拆分
#
# 量化分支：SCENARIO==QUANT 时，weight_uq_qr 为 INT8，需 cast + dequant；
#   QUANT_MODE==1 时，kv_cache/kr_cache 为 INT8 per-channel 量化。
# 主循环：CANN 的 Process 含 for 循环（每步一个 M 轴 chunk），此处简化为单块。
# 6 CV sync event（SYNC_MMCQ_NOMRCQ 等）不在此体现。
# grid = (ceil(token_num / BLOCK_T),) * N  # 多核沿 token×head