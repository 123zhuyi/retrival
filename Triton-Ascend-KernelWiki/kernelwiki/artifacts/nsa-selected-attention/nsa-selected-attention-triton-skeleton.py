# nsa_selected_attention - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# NSA 训练前向第二阶段：给定 topKIndices，gather 选中 KV 块并计算标准 attention。
# 来源：attention/nsa_selected_attention/ (commit d28d8f9...), docs L22-28.

import triton
import triton.language as tl


@triton.jit
def nsa_selected_attention_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, topk_indices_ptr,
    atten_mask_ptr,
    softmax_max_ptr, softmax_sum_ptr, out_ptr,
    actual_q_seqlen_ptr, actual_kv_seqlen_ptr,
    # scalars
    sm_scale,
    num_heads, kv_heads,
    selected_block_size: tl.constexpr,
    selected_block_count: tl.constexpr,
    selected_length: tl.constexpr,  # = selected_block_size * selected_block_count
    D_QK: tl.constexpr,  # 192
    D_V: tl.constexpr,   # 128
    BLOCK_M: tl.constexpr,
    CAUSAL: tl.constexpr,  # sparseMode=2
):
    pid = tl.program_id(0)
    # 任务 = (S1 row, kvHead)
    # ...

    # 1. 加载 topKIndices（本 Q 行选中的 KV 块索引）
    idx = tl.load(topk_indices_ptr + offset)
    selected_k_offs = idx * selected_block_size * D_QK + tl.arange(0, selected_block_size)[:, None] * D_QK + tl.arange(0, D_QK)[None, :]
    selected_v_offs = idx * selected_block_size * D_V + tl.arange(0, selected_block_size)[:, None] * D_V + tl.arange(0, D_V)[None, :]

    # 2. Gather 选中的 KV 块
    k = tl.load(k_ptr + selected_k_offs)
    v = tl.load(v_ptr + selected_v_offs)

    # 3. QK^T + scale
    s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale

    # 4. Mask（causal 时）
    if CAUSAL:
        s = tl.where(causal_mask, s, -float("inf"))

    # 5. 在线 softmax（running max/sum，输出 max/sum 供反向用）
    m_i = tl.max(s, axis=1)  # [BLOCK_M]
    p = tl.exp(s - m_i[:, None])
    l_i = tl.sum(p, axis=1)  # [BLOCK_M]

    # 6. softmaxMax/Sum 输出（供反向）
    tl.store(softmax_max_ptr, m_i)
    tl.store(softmax_sum_ptr, l_i)

    # 7. PV
    acc = tl.dot(p, v).to(tl.float32)
    # 8. 写回
    tl.store(out_ptr, acc.to(out_ptr.dtype.element_ty))

    # 注意：
    # - 仅 TND 布局（3D [T, N, D]）
    # - D_qk=192, D_v=128（固定）
    # - 仅 A2 训练场景
    # - softmaxMax/Sum shape [T, N1, 8]（每 G 组 reduce 宽度=8）
    pass