#!/usr/bin/env python3
# """
# QuantLightningIndexer - Triton-Ascend skeleton
#
# 知识制品 (knowledge artifact only)：表达设计意图，非可编译移植。
# 未编译、未在 NPU 运行、未 benchmark。reproducibility: pseudocode。
#
# 映射 CANN QuantLightningIndexer 公式（README.md L16-20）：
#   out = Top_k{ W @ [ ReLU((Scale_Q * Scale_K^T) * (Q^Quant @ K^Quant^T)) ] }
#
# 映射备忘：
#   - program_id(0) <- CANN S1 分核
#   - BLOCK_S1 <- CANN S1 基本块
#   - BLOCK_S2 <- S2 整块
#   - FP32 累加 <- CANN int8/fp8 反量化 FP32 计算
#   - PA_BSND block_table <- 索引查找离散 KV
# """
import triton
import triton.language as tl


@triton.jit
def _quant_lightning_indexer_kernel(
    q_ptr, k_ptr, weights_ptr, q_dequant_scale_ptr, k_dequant_scale_ptr,
    sparse_indices_ptr, block_table_ptr,
    stride_qb, stride_qn, stride_qs, stride_qd,
    stride_kb, stride_kn, stride_ks, stride_kd,
    stride_wb, stride_wn, stride_ws,
    stride_qdb, stride_qdn, stride_qds,
    stride_kdb, stride_kdn, stride_kds,
    S1, S2, G, N2, D,
    sparse_count: tl.constexpr,
    is_page_attention: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_S1: tl.constexpr,
    BLOCK_S2: tl.constexpr,
):
    pid = tl.program_id(0)  # S1 块索引

    offs_s1 = pid * BLOCK_S1 + tl.arange(0, BLOCK_S1)
    offs_d = tl.arange(0, D)
    mask_s1 = offs_s1 < S1

    # Q [BLOCK_S1, G, D] 量化 int8/fp8，反量化到 FP32
    q = tl.load(q_ptr + offs_s1[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=mask_s1[:, None], other=0.0).to(tl.float32)
    q_scale = tl.load(q_dequant_scale_ptr + offs_s1 * stride_qds,
                       mask=mask_s1, other=0.0).to(tl.float32)

    # K [BLOCK_S2, N2, D] 量化 int8/fp8，反量化到 FP32
    k = tl.load(k_ptr + offs_d[None, :] * stride_kd,
                mask=mask_s1[:, None], other=0.0).to(tl.float32)
    k_scale = tl.load(k_dequant_scale_ptr, other=0.0).to(tl.float32)

    # 反量化 QK^T
    qk = tl.dot(q * q_scale[:, None], tl.trans(k * k_scale))  # [BLOCK_S1, BLOCK_S2]

    # ReLU 过滤负相关
    scores = tl.maximum(0, qk)

    # 因果掩码 (sparse_mode=3)
    if is_causal:
        offs_s2 = tl.arange(0, BLOCK_S2)
        scores = tl.where(offs_s1[:, None] >= offs_s2[None, :], scores, -float("inf"))

    # W @ 相关性 (跨 G 轴加权聚合)
    w = tl.load(weights_ptr + offs_s1 * stride_ws, mask=mask_s1, other=0.0).to(tl.float32)
    agg_scores = tl.sum(scores * w[:, None], axis=0)  # [BLOCK_S2]

    # topk 索引
    topk_vals, topk_idxs = tl.topk(agg_scores, sparse_count)
    tl.store(sparse_indices_ptr + pid * sparse_count + tl.arange(0, sparse_count),
             topk_idxs)


# 映射备忘：
# - program_id(0) <- CANN S1 分核
# - BLOCK_S1 <- CANN S1 基本块
# - BLOCK_S2 <- S2 整块
# - FP32 累加 <- CANN int8/fp8 反量化 FP32 计算
# - PA_BSND block_table <- 索引查找离散 KV（此处简化为连续 KV）
# - is_causal <- CANN sparse_mode=3