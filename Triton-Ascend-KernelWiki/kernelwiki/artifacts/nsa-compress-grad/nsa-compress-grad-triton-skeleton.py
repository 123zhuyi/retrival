# nsa_compress_grad - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：NSA compress 反向，纯 Vector 计算。
#   dw = dk_cmp · K^T（weight 梯度，沿 headDim reduce）
#   dk = W^T · dk_cmp（input 梯度，weight 广播后 mul）
#   支持变长 TND（actSeqLen cumsum）、overlap 累加（blockSize > blockStride）、AtomicAdd 规约。
# CANN 专有机制（MTE DataCopyPad / SetFlag-WaitFlag 同步 / workspace ping-pong / 确定性树形规约）不在此体现。
# 来源：attention/nsa_compress_grad/ (commit d28d8f9...), op_kernel/nsa_compress_grad.h L330-370.

import triton
import triton.language as tl


@triton.jit
def nsa_compress_grad_kernel(
    # pointers
    output_grad_ptr,        # [total_block_num, N, D] dk_cmp
    input_ptr,              # [T, N, D]             K
    weight_ptr,             # [compressBlockSize, N] W
    act_seq_len_ptr,        # cumsum per batch
    input_grad_ptr,         # [T, N, D]             dk（输出）
    weight_grad_ptr,        # [compressBlockSize, N] dw（输出）
    # scalars
    total_block_num,
    compress_block_size: tl.constexpr,
    compress_stride: tl.constexpr,
    batch_size,
    # block meta
    N: tl.constexpr,        # numHeads
    D: tl.constexpr,        # headDim
):
    pid = tl.program_id(0)  # block 维外切（CANN aivNum 分核）

    # 每 block 对应 output_grad 的一个位置 [block_id, head_id, dim]
    # output_grad[blk_id, :, :] → dk_cmp[blk_id, :, :]
    dk_cmp = tl.load(output_grad_ptr + pid * N * D + tl.arange(0, N * D)).to(tl.float32)  # [N, D]

    # 解析当前 block 对应的 input 起始位置（基于 cumsum actSeqLen + blockId × stride）
    # 伪代码：batch_offset, seq_offset = resolve_cumsum(act_seq_len_ptr, pid, compress_block_size, compress_stride)
    # K_block = input[batch_offset + seq_offset : batch_offset + seq_offset + compress_block_size, :, :]
    # 简化：此处假设已算出 input_offset
    input_offset = pid * compress_stride * N * D
    K_block = tl.load(input_ptr + input_offset + tl.arange(0, compress_block_size * N * D)).to(tl.float32)
    K_block = K_block.reshape([compress_block_size, N, D])  # [blockSize, N, D]

    # dw = dk_cmp · K^T → [blockSize, N]
    # 沿 D 维 dot 后 reduce
    dw = tl.dot(dk_cmp.reshape([N, D]), K_block.trans(1, 2, 0))  # [N, N, blockSize]
    dw = tl.sum(dw, axis=0)                                       # [N, blockSize]
    dw = dw.trans(1, 0)                                           # [blockSize, N]

    # AtomicAdd 写回 weight_grad（多核可能写同位置）
    tl.atomic_add(weight_grad_ptr + tl.arange(0, compress_block_size * N), dw.reshape([compress_block_size * N]))

    # dk = W^T · dk_cmp → [blockSize, N, D]
    W = tl.load(weight_ptr + tl.arange(0, compress_block_size * N)).to(tl.float32)  # [blockSize, N]
    dk = tl.dot(W.T, dk_cmp.reshape([N, D]))                                        # [blockSize, N, D] → [blockSize×N, D]
    # 简化：W.T 形状为 [N, blockSize]，dk_cmp 为 [N, D] → W·dk_cmp 需 broadcast
    # 实际语义：dk[block_row, n, d] = sum_over_head(W[block_row, n] * dk_cmp[block_id, n, d])
    # 即 weight 沿 D 维广播后与 dk_cmp 逐元素乘
    dk = W[:, :, None] * dk_cmp[None, :, :]                      # [blockSize, N, D]

    # AtomicAdd（overlap 场景）或直接 store
    # 非 overlap: dk 对应 input_offset 位置直接写
    tl.store(input_grad_ptr + input_offset + tl.arange(0, compress_block_size * N * D),
             dk.reshape([compress_block_size * N * D]))


# grid = (total_block_num,)   # 沿 block 维外切
# 多核写 weight_grad 同位置 → tl.atomic_add
# 变长（TND + actSeqLen cumsum）→ per-batch offset + mask