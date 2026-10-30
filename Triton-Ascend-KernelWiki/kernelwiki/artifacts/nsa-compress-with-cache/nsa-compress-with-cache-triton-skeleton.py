# nsa_compress_with_cache - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：NSA KV 压缩推理阶段（compressBlockSize 滑动窗口加权求和 + scatter 写回）。
# 来源：attention/nsa_compress_with_cache/ (commit d28d8f9...), README L17-29.

import triton
import triton.language as tl


@triton.jit
def nsa_compress_with_cache_kernel(
    input_ptr, weight_ptr, slot_mapping_ptr, output_cache_ptr,
    act_seq_len_ptr, block_table_ptr,
    # shape
    batch_size, head_num, head_dim, kv_cache_size,
    # meta (constexpr) -- 值 source-reported
    COMPRESS_BLOCK_SIZE: tl.constexpr,  # 压缩滑窗（16 倍数，≤64）
    COMPRESS_STRIDE: tl.constexpr,      # 间隔（16 倍数）
    BLOCK_HEAD: tl.constexpr,           # 每核 head 数
    PAGE_BLOCK_SIZE: tl.constexpr,      # 64 or 128
):
    # 每核负责 head×batch 范围
    pid = tl.program_id(0)
    batch_idx = pid // (head_num // BLOCK_HEAD)
    head_start = (pid % (head_num // BLOCK_HEAD)) * BLOCK_HEAD
    offs_head = head_start + tl.arange(0, BLOCK_HEAD)
    h_mask = offs_head < head_num

    # 检查压缩条件
    seq_len = tl.load(act_seq_len_ptr + batch_idx)
    cond = (seq_len >= COMPRESS_BLOCK_SIZE) and ((seq_len - COMPRESS_BLOCK_SIZE) % COMPRESS_STRIDE == 0)
    if not cond:
        return

    compress_idx = (seq_len - COMPRESS_BLOCK_SIZE) // COMPRESS_STRIDE
    offs_token = compress_idx * COMPRESS_STRIDE + tl.arange(0, COMPRESS_BLOCK_SIZE)
    offs_dim = tl.arange(0, head_dim)
    offs_head_dim = offs_head[:, None] * head_dim + offs_dim[None, :]

    # 滑动窗口 token × weight -> 加权求和
    # input: [batch, seq, head, dim] 压缩后 compress_block_size 个 token
    inp = tl.load(input_ptr + batch_idx * kv_cache_size
                  + offs_token[:, None, None] * head_num * head_dim
                  + offs_head_dim[None, :, :],
                  mask=(offs_token[:, None, None] < seq_len) & h_mask[None, :, None],
                  other=0.0).to(tl.float32)  # [COMPRESS_BLOCK_SIZE, BLOCK_HEAD, head_dim]
    weight = tl.load(weight_ptr + tl.arange(0, COMPRESS_BLOCK_SIZE)).to(tl.float32)  # [COMPRESS_BLOCK_SIZE]
    compressed = tl.sum(inp * weight[:, None, None], axis=0)  # [BLOCK_HEAD, head_dim]

    # scatter 写回（slot_mapping 索引）
    slot = tl.load(slot_mapping_ptr + batch_idx)
    tl.store(output_cache_ptr + slot * head_num * head_dim + offs_head_dim,
             compressed, mask=h_mask[:, None])


# 注意：pageBlockSize 场景下，需通过 block_table 查物理 page 偏移（PA 场景）。
# grid = (batch_size * (head_num // BLOCK_HEAD),)