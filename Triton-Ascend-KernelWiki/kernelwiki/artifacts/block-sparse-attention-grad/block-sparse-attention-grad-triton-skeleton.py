# block_sparse_attention_grad - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：块稀疏注意力反向 -- mask 门控稠密扫描 + 重算 P（用前向 Lse）
#   + softmax-grad + dQ/dK/dV（dK/dV 跨程序 atomic 累加）+ GQA。
# CANN 专有机制（MTE/BufferID/CrossCore event/preTaskInfo delay/ND↔NZ/TQue/AtomicAdd）不在此体现。
# 来源：attention/block_sparse_attention_grad/ (commit d28d8f9...), docs L19-71, README L19-71.
#
# 映射注记（见 artifact tiling_dispatch_tables.md §C）：
#   program_id <- CANN core 沿 (batch, head, Q-block)（无 FlashDecoding，KV 不切核）
#   BLOCK_M <- BASIC_BLOCK_SIZE=128 (source-reported)
#   mask 门控扫描 <- CANN kernel.h L386-389,604-607（if(mask) skip，非前向 CSR gather）
#   重算 P <- CANN SimpltSoftmax: P=exp(S·scale−Lse)（用前向保存 Lse）
#   dK/dV atomic <- CANN block_mmad_bsag2 Cube3 isAtomicAdd（跨 Q-block 累加同 KV）
#   rescale <- CANN BlockPost: dQ/dK*=scale, dV 不乘

import triton
import triton.language as tl


@triton.jit
def block_sparse_attention_grad_kernel(
    # pointers
    dout_ptr, q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    block_sparse_mask_ptr,            # [B, N1, qBlockNum, kvBlockNum] bool（反向直接读原始 mask）
    actual_qseqlen_ptr, actual_kvseqlen_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    # shape / scalars
    sm_scale,
    num_heads, kv_heads,
    q_block_num, kv_block_num,
    # feature flags (constexpr)
    Q_LAYOUT_TND: tl.constexpr,       # True=TND, False=BNSD
    # block meta (constexpr) -- 值 source-reported
    BLOCK_M: tl.constexpr,            # BASIC_BLOCK_SIZE=128（Q 方向）
    BLOCK_N: tl.constexpr,            # blockShapeY（KV 方向，128 倍数；占位 128）
    D: tl.constexpr,                  # headDim=128
    BLOCKSHAPE_X: tl.constexpr,
    BLOCKSHAPE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    # 任务 = (batch, qHead, qBlock)；同前向 totalTaskNum = Q-block×head
    qblk_id = pid % q_block_num
    head_id = (pid // q_block_num) % num_heads
    batch_id = pid // (q_block_num * num_heads)

    # GQA：KV head = qHead / groupSize
    group_size = num_heads // kv_heads
    kv_head_id = head_id // group_size

    # Q / dout 行范围（含 tail）
    start_m = qblk_id * BLOCKSHAPE_X
    offs_m = start_m + tl.arange(0, BLOCK_M)
    q_len = tl.load(actual_qseqlen_ptr + batch_id)
    m_mask = offs_m < q_len
    offs_d = tl.arange(0, D)

    q = tl.load(q_ptr + q_addr(batch_id, head_id, offs_m, offs_d, Q_LAYOUT_TND),
                mask=m_mask[:, None], other=0.0).to(tl.float32)
    dy = tl.load(dout_ptr + q_addr(batch_id, head_id, offs_m, offs_d, Q_LAYOUT_TND),
                 mask=m_mask[:, None], other=0.0).to(tl.float32)
    o = tl.load(out_ptr + q_addr(batch_id, head_id, offs_m, offs_d, Q_LAYOUT_TND),
                mask=m_mask[:, None], other=0.0).to(tl.float32)
    # 前向保存的 Lse（每 Q 行一个）
    lse = tl.load(lse_ptr + lse_addr(batch_id, head_id, offs_m, Q_LAYOUT_TND), mask=m_mask, other=0.0)

    # softmax-grad 归一化项 D = Σ(dY · O)（每 Q 行）-- CANN SoftmaxGrad
    d_row = tl.sum(dy * o, axis=1)                       # [BLOCK_M]

    # dQ 累加器（核内跨 KV 块累加）
    dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # ---- mask 门控稠密扫描：遍历所有 KV 块，mask=0 跳过（CANN if(mask) skip）----
    for kvblk in range(0, kv_block_num):
        # 读原始 bool mask（反向不复用前向 CSR）
        mask_val = tl.load(block_sparse_mask_ptr +
                           mask_addr(batch_id, head_id, qblk_id, kvblk, num_heads, q_block_num, kv_block_num))
        if mask_val == 0:
            continue                                      # 该 KV 块不参与

        kv_start = kvblk * BLOCKSHAPE_Y
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        kv_len = tl.load(actual_kvseqlen_ptr + batch_id)
        n_mask = offs_n < kv_len

        k = tl.load(k_ptr + kv_addr(batch_id, kv_head_id, offs_n, offs_d, Q_LAYOUT_TND),
                    mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr + kv_addr(batch_id, kv_head_id, offs_n, offs_d, Q_LAYOUT_TND),
                    mask=n_mask[:, None], other=0.0)

        # S = Q·K^T * scale；重算 P = exp(S - Lse)（CANN SimpltSoftmax）
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        s = tl.where(m_mask[:, None] & n_mask[None, :], s, -float("inf"))
        p = tl.exp(s - lse[:, None])                      # [BLOCK_M, BLOCK_N]

        # dP = dY·V^T
        dp = tl.dot(dy, tl.trans(v)).to(tl.float32)       # [BLOCK_M, BLOCK_N]
        # dS = P * (dP - D)
        ds = p * (dp - d_row[:, None])                     # [BLOCK_M, BLOCK_N]

        # dQ += dS·K  （核内累加，FP32）
        dq_acc += tl.dot(ds, k).to(tl.float32)            # [BLOCK_M, D]

        # dK += dS^T·Q  （跨 Q-block 累加同 KV -> atomic_add 到 GM）
        dk = tl.dot(tl.trans(ds), q).to(tl.float32) * sm_scale   # [BLOCK_N, D]
        tl.atomic_add(dk_ptr + kv_addr(batch_id, kv_head_id, offs_n, offs_d, Q_LAYOUT_TND),
                      dk, mask=n_mask[:, None])

        # dV += P^T·dY  （跨 Q-block 累加同 KV -> atomic_add；dV 不乘 scale）
        dv = tl.dot(tl.trans(p), dy).to(tl.float32)       # [BLOCK_N, D]
        tl.atomic_add(dv_ptr + kv_addr(batch_id, kv_head_id, offs_n, offs_d, Q_LAYOUT_TND),
                      dv, mask=n_mask[:, None])

    # rescale 写回：dQ *= scale（CANN BlockPost：dQ/dK 乘 scale，dV 不乘；dK 已在上面乘）
    dq_acc = dq_acc * sm_scale
    dq = dq_acc.to(dq_ptr.dtype.element_ty)
    tl.store(dq_ptr + q_addr(batch_id, head_id, offs_m, offs_d, Q_LAYOUT_TND), dq, mask=m_mask[:, None])


# 辅助（伪）：q_addr / kv_addr / lse_addr / mask_addr 表意 TND vs BNSD 的地址算术，
#   对应 CANN kernel.h L224,258-260,388,606 的 batch/head/qblk/kvblk offset 计算。
#
# grid = (totalTaskNum,)  # = sum_b ceilDiv(qseqlen_b, blockShapeX) * numHead   -- 与 CANN 一致
#
# 设计注记：
#  - 反向 mask 处理 = 门控稠密扫描（if mask==0: continue），与前向 CSR 压缩 + gather 不同；
#    Triton 侧也可改用前向的 selectIdx gather（见 [[technique-block-sparse-mask-gather]]），
#    两种策略的取舍（CSR 一次性压缩 vs 逐块分支）需结合稀疏度实测 [needs_verification]。
#  - dQ 核内 FP32 累加（一个 program 拥有一个 Q-block）；dK/dV 跨 program 累加同 KV -> atomic_add。
#  - 重算 P = exp(S·scale - Lse)，用前向保存的 lse；attentionOut 用于 softmax-grad 项 D=Σ(dY·O)。
#  - rescale：dQ/dK 乘 scale，dV 不乘（CANN BlockPost L371-374）。
#  - 反向 softmax 内部恒 FP32（无前向 innerPrecise fp16 路径）。
#  - 仅训练场景；无 PagedAttention/滑窗/causal（maskType 仅 0）。
