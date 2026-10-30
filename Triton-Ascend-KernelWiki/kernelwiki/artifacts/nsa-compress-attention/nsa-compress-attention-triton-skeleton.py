#!/usr/bin/env python3
# """
# NsaCompressAttention - Triton-Ascend skeleton
#
# 知识制品 (knowledge artifact only)：表达设计意图，非可编译移植。
# 未编译、未在 NPU 运行、未 benchmark。reproducibility: pseudocode。
#
# 映射 CANN NsaCompressAttention 公式（README.md L16-38）：
#   1. P_cmp = Softmax(Q * K^T)
#   2. attention_out = Softmax(mask(scale * Q * K^T, atten_mask)) * V
#   3. P_slc[j] = sum_{m=0}^{l'/d-1} sum_{n=0}^{l/d-1} P_cmp[l'/d*j - m - n]
#   4. P_slc' = sum_{h=1}^{H} P_slc^h
#   5. P_slc' = topk_mask(P_slc')
#   6. topkIndices = topk(P_slc')
#
# CANN -> Triton-Ascend 映射：
#   - 核间切分 (B,N2,S1_block) -> grid + tl.program_id
#   - S1基本块/CV_RATIO           -> tl.constexpr BLOCK_M
#   - S2 不做切分 (整块)          -> BLOCK_N = S2
#   - FP32 累加                   -> tl.dot(..).to(tl.float32)
#   - importance score 滑动窗口   -> 简单循环累加
#   - 跨 head 累加                -> tl.sum 跨 G 轴
#   - topk 索引                   -> tl.topk (或手工排序)
#   - MTE/BufferID/queue/event   -> 抽象为 tile 级 fused load/store
#   - 仅支持 TND layout
# """
import triton
import triton.language as tl


@triton.jit
def _nsa_compress_attn_kernel(
    q_ptr, k_ptr, v_ptr,
    atten_mask_ptr, topk_mask_ptr,
    attn_out_ptr, softmax_max_ptr, softmax_sum_ptr, topk_indices_ptr,
    sm_scale,
    stride_qb, stride_qn, stride_qs, stride_qd,
    stride_kb, stride_kn, stride_ks, stride_kd,
    stride_vb, stride_vn, stride_vs, stride_vd,
    stride_attnb, stride_attnn, stride_attns, stride_attnd,
    Sq, Skv,
    compress_block_size: tl.constexpr,   # l
    compress_stride: tl.constexpr,       # d
    select_block_size: tl.constexpr,    # l'
    select_block_count: tl.constexpr,   # n
    BLOCK_M: tl.constexpr,   # = S1基本块 (CANN: 128/gSize)
    BLOCK_N: tl.constexpr,   # = Skv (整块)
    D: tl.constexpr,         # head dim
    DV: tl.constexpr,        # value head dim
    G: tl.constexpr,         # query/kv head 比
    HAS_MASK: tl.constexpr,
    HAS_TOPK_MASK: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # n2
    pid_m = tl.program_id(2)  # S1 块索引

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_dv = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < Sq
    mask_n = offs_n < Skv

    # Q [BLOCK_M, D], K [BLOCK_N, D], V [BLOCK_N, DV]
    q = tl.load(q_ptr + pid_b * stride_qb + pid_n * stride_qn + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=mask_m[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_ptr + pid_b * stride_kb + pid_n * stride_kn + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptr + pid_b * stride_vb + pid_n * stride_vn + offs_n[:, None] * stride_vs + offs_dv[None, :] * stride_vd,
                mask=mask_n[:, None], other=0.0).to(tl.float32)

    # 步 1: P_cmp = Softmax(Q * K^T)  (压缩注意力)
    s = tl.dot(q, tl.trans(k)) * sm_scale
    if HAS_MASK:
        mask = tl.load(atten_mask_ptr + pid_b * stride_attnb + pid_n * stride_attnn + offs_m[:, None] * stride_attns + offs_n[None, :] * stride_attnd,
                        mask=mask_m[:, None] & mask_n[None, :])
        s = tl.where(mask, -float("inf"), s)

    # 在线 softmax (running max/sum + rescale)
    m_i = tl.max(s, axis=1)
    p = tl.exp(s - m_i[:, None])
    l_i = tl.sum(p, axis=1)
    p_norm = p / l_i[:, None]  # 归一化 P

    # 写回 softmax_max/sum 供反向
    tl.store(softmax_max_ptr + pid_b * Sq + pid_n * Sq * G + offs_m, m_i, mask=mask_m)
    tl.store(softmax_sum_ptr + pid_b * Sq + pid_n * Sq * G + offs_m, l_i, mask=mask_m)

    # 步 2: attention_out = Softmax(mask(scale * QK^T)) * V
    attn_out = tl.dot(p_norm.to(v.dtype), v)  # [BLOCK_M, DV]
    tl.store(attn_out_ptr + pid_b * stride_attnb + pid_n * stride_attnn + offs_m[:, None] * stride_attns + offs_dv[None, :] * stride_attnd,
             attn_out.to(attn_out_ptr.dtype.element_ty), mask=mask_m[:, None])

    # 步 3: importance score 滑动窗口 (简化版 - 实际需更细致的 sliding window)
    # P_slc[j] = sum_{m=0}^{l'/d-1} sum_{n=0}^{l/d-1} P_cmp[l'/d * j - m - n]
    is_m = select_block_size // compress_stride
    is_n = compress_block_size // compress_stride
    s2_per_score = is_m + is_n - 1
    imp_score_blocks = tl.cdiv(Skv, is_m * BLOCK_M)  # 简化的块数
    # 注：完整实现需按 sliding window 遍历并跨 head 累加

    # 步 4: 跨 head 累加 P_slc' = sum_h P_slc^h (简化: 跨 G 轴 reduce)
    imp_score = tl.zeros([BLOCK_M, imp_score_blocks], dtype=tl.float32)
    # ... (滑动窗口收集 + tl.sum 跨 G)

    # 步 5: topk_mask
    if HAS_TOPK_MASK:
        # ... (tl.where)

    # 步 6: topk 索引 (简化: 假设 topk 值可用)
    # tl.topk(imp_score, select_block_count) 或手工排序
    topk_vals, topk_idxs = tl.topk(imp_score, select_block_count)
    tl.store(topk_indices_ptr + pid_b * ... + pid_n * ... + offs_m * select_block_count + tl.arange(0, select_block_count),
             topk_idxs, mask=mask_m[:, None])


# 映射备忘：
# - program_id(0/1/2) <- CANN 核间切 (B, N2, S1_block)
# - BLOCK_M <- CANN s1BasicBlock = NSA_BASE_S1G_SIZE / gSize (128/gSize)
# - BLOCK_N <- CANN s2BasicBlock = alignedS2 (整块)
# - FP32 累加 <- CANN L0C/UB FP32 中间
# - importance score 滑动窗口 <- CANN ProcessVec2 复杂 UB 流水
# - topk 索引 <- CANN TopK API
# - HAS_MASK/HAS_TOPK_MASK <- CANN tiling key bit 字段