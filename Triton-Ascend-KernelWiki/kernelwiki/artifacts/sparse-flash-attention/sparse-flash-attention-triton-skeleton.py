# sparse_flash_attention - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：稀疏注意力推理，sparse_indices 驱动离散 KV 取数 + 在线 softmax + MLA-absorb。
# CANN 专有机制（MTE/BufferID/CrossCore event/PreloadPipeline/ND↔NZ）不在此体现。
# 来源：attention/sparse_flash_attention/ (commit d28d8f9...), README L16-20/73.
#
# 映射注记（见 artifact tiling_dispatch_tables.md §C）：
#   program_id <- CANN core 沿 B/N2/G/S1（FlashDecoding 时 +S2）
#   BLOCK_M <- mBaseSize=128 (source-reported)
#   BLOCK_N <- s2BaseSize 动态（source-reported）
#   sparse_indices 驱动 S2 <- CANN CalcSinnerTopKBegin 离散 block 遍历
#   MLA D=576 <- nope(512)+rope(64) 沿 D 拼接

import triton
import triton.language as tl


@triton.jit
def sparse_flash_attention_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, o_ptr, sm_max_ptr, sm_sum_ptr,
    sparse_indices_ptr,          # [B, Q_S, KV_N, sparse_size] int32 - 离散 KV block 索引
    block_table_ptr,             # PagedAttention（可选）
    query_rope_ptr, key_rope_ptr,
    # strides (省略)
    # shape / scalars
    sm_scale,
    q_len, sparse_size,
    # feature flags (constexpr)
    USE_PA: tl.constexpr,        # PagedAttention
    MLA: tl.constexpr,           # attention_mode=2
    FLASH_DECODE: tl.constexpr,  # S2 核间切分
    RETURN_LSE: tl.constexpr,
    # block meta (constexpr) -- 值 source-reported
    BLOCK_M: tl.constexpr,       # mBaseSize=128
    BLOCK_N: tl.constexpr,       # s2BaseSize（动态，此处占位）
    D_NOPE: tl.constexpr,        # 512
    D_ROPE: tl.constexpr,        # 64
):
    pid = tl.program_id(0)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < q_len

    # MLA-absorb: Q/K 沿 D 维拼接 nope + rope -> D=576
    offs_d_nope = tl.arange(0, D_NOPE)
    offs_d_rope = tl.arange(0, D_ROPE)
    if MLA:
        q_nope = tl.load(q_ptr + offs_m[:, None] * D_NOPE + offs_d_nope[None, :],
                         mask=m_mask[:, None], other=0.0)
        q_rope = tl.load(query_rope_ptr + offs_m[:, None] * D_ROPE + offs_d_rope[None, :],
                         mask=m_mask[:, None], other=0.0)
        q = tl.cat(q_nope, q_rope)   # [BLOCK_M, 576]（伪：tl.cat 表意）
    else:
        q = tl.load(q_ptr + offs_m[:, None] * D_NOPE + offs_d_nope[None, :],
                    mask=m_mask[:, None], other=0.0).to(tl.float32)

    # 在线 softmax 状态
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_NOPE], dtype=tl.float32)   # V 仅 nope 部分（DV=512）

    # ---- 稀疏 S2 遍历：由 sparse_indices 驱动（CANN CalcSinnerTopKBegin）----
    # 每次从 sparse_indices 读取一批 block 索引，聚合为 BLOCK_N 大小的连续 chunk
    for s_idx in range(0, sparse_size, BLOCK_N):
        # 1. 读 sparse_indices 得到 KV block 索引（-1 表无效，跳过）
        idx = tl.load(sparse_indices_ptr + offs_m * sparse_size + s_idx + tl.arange(0, BLOCK_N),
                      mask=m_mask[:, None], other=-1)   # [BLOCK_M, BLOCK_N]（简化）
        valid = idx >= 0

        # 2. 离散 KV gather（CANN MergeKv：离散 block -> 连续 chunk）
        #    PagedAttention 时先 block_table 查物理 block，再取 KV
        if USE_PA:
            phys = tl.load(block_table_ptr + idx)        # 块表查物理 block（伪）
            kv_offs = phys * BLOCK_N + tl.arange(0, BLOCK_N)
        else:
            kv_offs = idx * BLOCK_N + tl.arange(0, BLOCK_N)

        k = tl.load(k_ptr + kv_offs[:, None] * D_NOPE + offs_d_nope[None, :])  # [BLOCK_N, 512]
        if MLA:
            k_rope = tl.load(key_rope_ptr + kv_offs[:, None] * D_ROPE + offs_d_rope[None, :])
            k = tl.cat(k, k_rope)                        # [BLOCK_N, 576]
        v = tl.load(v_ptr + kv_offs[:, None] * D_NOPE + offs_d_nope[None, :])  # [BLOCK_N, 512]

        # 3. QK^T + scale + 稀疏 mask（无效 block -> -inf）+ rightDownCausal
        s = tl.dot(q, k.trans(1, 0)).to(tl.float32) * sm_scale
        s = tl.where(valid, s, -float("inf"))            # 无效 sparse block 遮蔽
        # sparse_mode=3 rightDownCausal：下三角 mask（略，按 offs_m/kv_offs 关系）

        # 4. 在线 FlashSoftmax（running max/sum + exp rescale + PV 累加）
        m_ij = tl.max(s, axis=1)
        s = s - m_ij[:, None]
        p = tl.exp(s).to(v.dtype)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p, v).to(tl.float32)
        m_i = tl.maximum(m_i, m_ij)

    # rescale 写回
    acc = acc / l_i[:, None]
    o = acc.to(o_ptr.dtype.element_ty())
    tl.store(o_ptr + offs_m[:, None] * D_NOPE + offs_d_nope[None, :], o, mask=m_mask[:, None])

    # 可选：输出 softmax_max / softmax_sum（return_softmax_lse，训练/非 PA 场景）
    if RETURN_LSE:
        tl.store(sm_max_ptr + offs_m, m_i, mask=m_mask)
        tl.store(sm_sum_ptr + offs_m, l_i, mask=m_mask)


# C_TEMPLATE vs V_TEMPLATE：V_TEMPLATE 额外做 MergeKv（Vector 侧离散->连续聚合），
#   触发条件 sparseBlockSize≤4 && s2Size≠0。Triton 层可统一为“先 gather 再 dot”，
#   C/V 差异主要体现在 gather 粒度与流水，Triton 不显式 CV 分离。
# FlashDecoding：S2 核间切分时，各核独立计算局部 max/sum/PV，最后跨核 reduce 合并
#   （合并公式同 online softmax：alpha=exp(m_a-m_merged)，rescale 累加）。
# grid = (ceil(q_len / BLOCK_M) * B * N2 * G, [, splitKV_num])  # FlashDecoding 时二维
