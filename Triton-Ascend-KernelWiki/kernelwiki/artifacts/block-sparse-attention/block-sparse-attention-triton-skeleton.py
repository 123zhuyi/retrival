# block_sparse_attention - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：块稀疏注意力 —— blockSparseMask 压缩为选中 KV-block 索引 -> 稀疏 gather K/V
#   + 在线 softmax + GQA。QK->online softmax->PV 骨架同 FlashAttention。
# CANN 专有机制（MTE/BufferID/CrossCore event/PreloadPipeline/ND↔NZ/TQue）不在此体现。
# 来源：attention/block_sparse_attention/ (commit d28d8f9...), docs L18-22, README L123-146.
#
# 映射注记（见 artifact tiling_dispatch_tables.md §C）：
#   program_id <- CANN core 沿 (batch, head, Q-block)（无 FlashDecoding，KV 不切核）
#   BLOCK_M <- BASIC_BLOCK_SIZE=128 (source-reported)
#   blockSparseMask -> selectIdx 压缩 <- CANN Mask2IdxAndCount (kernel.h L93-248)
#   稀疏 KV gather <- CANN block_mmad_qk/pv 沿 selectIdx load K/V
#   FP32/Fp16 softmax <- innerPrecise 0/1 (FLOATSM/HALFSM)
#   GQA groupSize <- numHeads/numKvHeads，KV head 共享

import triton
import triton.language as tl


# ---- 第 0 步（host 或独立 kernel）：把 INT8 blockSparseMask 压缩为 CSR 式索引 ----
# CANN 在 vector 核全局执行 Mask2IdxAndCount 一次，产出：
#   selectIdx[b, h, qblk, :]   = 该 (b,h,qblk) 选中的 KV-block 索引（int32，变长，前置有效）
#   selectNum[b, h, qblk]      = 选中个数（int32）
# Triton 侧等价：host 预处理 mask -> selectIdx/selectNum 张量，或用 argwhere 风格 kernel。
# 本 skeleton 假设 selectIdx/selectNum 已就绪（与 CANN workspace 布局一致）。


@triton.jit
def block_sparse_attention_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    block_sparse_mask_ptr,         # [B, N1, qBlockNum, kvBlockNum] int8  （原始 mask，供 fallback）
    select_idx_ptr,                # [B, N1, qBlockNum, maxKvBlockNum] int32  选中 KV-block 索引
    select_num_ptr,                # [B, N1, qBlockNum] int32  选中个数
    actual_qseqlen_ptr,            # TND 必传；BNSD 可省
    actual_kvseqlen_ptr,
    # shape / scalars
    sm_scale,
    num_heads, kv_heads,
    q_block_num,                   # ceilDiv(maxQSeqlen, blockShapeX) * ...（按 batch）
    kv_block_num,                  # ceilDiv(maxKvSeqlen, blockShapeY)
    # feature flags (constexpr)
    Q_LAYOUT_TND: tl.constexpr,    # True=TND, False=BNSD
    LSE_OUT: tl.constexpr,         # softmaxLseFlag
    FP32_SOFTMAX: tl.constexpr,    # innerPrecise=0 -> True; =1 -> False
    # block meta (constexpr) -- 值 source-reported
    BLOCK_M: tl.constexpr,         # BASIC_BLOCK_SIZE=128（Q 方向）
    BLOCK_N: tl.constexpr,         # blockShapeY（KV 方向，128 倍数；占位 128）
    D: tl.constexpr,               # headDim ∈ {64, 128}
    BLOCKSHAPE_X: tl.constexpr,    # blockShapeX（Q 方向块大小）
    BLOCKSHAPE_Y: tl.constexpr,    # blockShapeY（= BLOCK_N，128 倍数）
    MAX_KV_BLOCK: tl.constexpr,    # selectIdx 末维大小
):
    pid = tl.program_id(0)
    # 任务 = (batch, qHead, qBlock)；与 CANN totalTaskNum = Q-block×head 同构
    qblk_id = pid % q_block_num
    head_id = (pid // q_block_num) % num_heads
    batch_id = pid // (q_block_num * num_heads)

    # GQA：KV head = qHead / groupSize（N1%N2==0）
    group_size = num_heads // kv_heads
    kv_head_id = head_id // group_size

    # Q 行范围（含 tail）
    start_m = qblk_id * BLOCKSHAPE_X
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # TND: q_ptr[totalQ, N1, D]；BNSD: q_ptr[B, N1, S, D] —— 此处以 TND 表意，BNSD 类似
    q_len = tl.load(actual_qseqlen_ptr + batch_id)
    m_mask = offs_m < q_len
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + (batch_id_offset(batch_id, Q_LAYOUT_TND)) + offs_m[:, None] * D + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)
    if not FP32_SOFTMAX:
        q = q.to(tl.float16)
    else:
        q = q.to(tl.float32)

    # 该 (batch, head, qblk) 选中的 KV-block 列表（CANN Mask2IdxAndCount 产物）
    sel_base = (batch_id * num_heads + head_id) * q_block_num + qblk_id
    sel_count = tl.load(select_num_ptr + sel_base)          # int32
    # 若整行 mask 全 0 -> 跳过该 Q-block（CANN: curSelectNum==0 continue）
    if sel_count == 0:
        o = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :], o.to(o_ptr.dtype.element_ty),
                 mask=m_mask[:, None])
        if LSE_OUT:
            tl.store(lse_ptr + offs_m, tl.full([BLOCK_M], -float("inf"), dtype=tl.float32), mask=m_mask)
        return

    # 在线 softmax 状态
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # ---- 稀疏 KV 遍历：沿 selectIdx gather，仅计算选中块（CANN block_mmad_qk/pv）----
    for s in range(0, MAX_KV_BLOCK):                # 上界为 selectIdx 末维
        if s >= sel_count:
            break                                    # 变长：超出有效个数即停
        kv_blk = tl.load(select_idx_ptr + sel_base * MAX_KV_BLOCK + s)   # 选中的 KV-block 索引
        kv_start = kv_blk * BLOCKSHAPE_Y
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        kv_len = tl.load(actual_kvseqlen_ptr + batch_id)
        n_mask = offs_n < kv_len

        # gather K / V（仅选中块；CANN copyGmToL1B 沿 selectIdx）
        k = tl.load(k_ptr + kv_head_offset(kv_head_id, Q_LAYOUT_TND) + offs_n[:, None] * D + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr + kv_head_offset(kv_head_id, Q_LAYOUT_TND) + offs_n[:, None] * D + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)

        # QK^T + scale（FP32 累加）
        s_ij = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        s_ij = tl.where(m_mask[:, None] & n_mask[None, :], s_ij, -float("inf"))

        # 在线 FlashSoftmax（running max/sum + exp rescale）
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s_ij - m_new[:, None])
        if not FP32_SOFTMAX:
            p = p.to(tl.float16)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v).to(tl.float32)
        m_i = m_new

    # rescale 写回
    acc = acc / l_i[:, None]
    o = acc.to(o_ptr.dtype.element_ty)
    tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :], o, mask=m_mask[:, None])

    # 可选：输出 softmaxLse = m_i + ln(l_i)（CANN rescale-O epilogue OUT_ONLY）
    if LSE_OUT:
        lse = m_i + tl.log(l_i)
        tl.store(lse_ptr + lse_offset(batch_id, head_id, qblk_id, Q_LAYOUT_TND) + offs_m,
                 lse, mask=m_mask)


# 辅助（伪）：batch_id_offset / kv_head_offset / lse_offset 表意 TND vs BNSD 的地址算术，
#   对应 CANN kernel.h L535,542 的 lse GM offset 与 GEMM 的 batch/head offset 计算。
#
# grid = (totalTaskNum,)  # = sum_b ceilDiv(qseqlen_b, blockShapeX) * numHeads   —— 与 CANN 一致
#
# 设计注记：
#  - CANN 的 mask 压缩（Mask2IdxAndCount）在 vector 核全局执行一次，Triton 侧建议 host 预处理
#    或独立 kernel 产出 selectIdx/selectNum，避免每个 program 重复压缩。
#  - 变长 selectIdx 在 Triton 静态 kernel 中用上界循环 + `if s>=sel_count: break` 表意；
#    实际实现可用 padding 索引 + mask，或按 Q-block 的最大选中数对齐。
#  - innerPrecise=1（FP16 softmax）路径：q/k/v/p 全 fp16，`tl.dot(..., out_dtype=tl.float16)`，
#    性能更佳但可能溢出，仅 FP16 输入可用（BF16 必须 FP32，即 FP32_SOFTMAX=True）。
#  - 无 FlashDecoding、无 PagedAttention、无滑窗、无 causal mask（maskType 仅 0）—— 与 CANN 一致。
