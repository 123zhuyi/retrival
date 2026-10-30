# nsa_compress_attention_infer - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：NSA 压缩注意力推理 —— PagedAttention KV + 滑动窗口压缩 +
#   importance score 卷积 + topK 选择。
# CANN 专有机制（MTE/BufferID/CrossCore event/双缓冲/ND↔NZ）不在此体现。
# 来源：attention/nsa_compress_attention_infer/ (commit d28d8f9...), docs L22-28.

import triton
import triton.language as tl


@triton.jit
def nsa_compress_attention_infer_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, output_ptr, topk_ptr,
    block_table_ptr,                    # [B, maxBlockNumPerBatch] int32
    actual_q_seqlen_ptr, actual_cmp_kv_seqlen_ptr, actual_sel_kv_seqlen_ptr,
    # shape / scalars
    sm_scale,
    num_heads, kv_heads,
    # NSA 参数
    compress_block_size: tl.constexpr,  # 滑窗大小
    compress_block_stride: tl.constexpr, # 滑窗步长
    select_block_size: tl.constexpr,    # 选择块大小
    select_block_count: tl.constexpr,   # topK 保留块数
    page_block_size: tl.constexpr,      # PagedAttention block 大小
    # block meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_QK: tl.constexpr,                 # QK head dim
    D_V: tl.constexpr,                  # V head dim
):
    pid = tl.program_id(0)
    # 任务 = (batch, kvHead, Q-token)
    # ... （grid 分配逻辑）

    # 1. QK^T — 通过 blockTable 映射读取压缩后的 KV
    #    遍历所有 KV block（blockTable 转换后）
    #    产生完整 score 矩阵

    # 2. Softmax — 在线 softmax（可复用 FA pipeline）

    # 3. PV — 加权求和得到 attentionOut

    # 4. 重要性分数 P_slc — 滑动窗口卷积
    #    三角形权重：中心高，两边低
    #    P_slc[j] = sum over i of weight[i] * score[i]

    # 5. topK 选择 — 取最重要的 selectBlockCount 个 KV block
    #    输出索引到 topk_ptr

    # 注意：
    # - KV 是 3D [numBlocks, blockSize, N*D] 格式，必须通过 blockTable 间接访问
    # - compressBlockSize/compressBlockStride 控制压缩滑窗
    # - selectBlockCount 是 topK 选择的块数
    # - 双输出：attentionOut + topKIndices
    pass