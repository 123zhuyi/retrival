# ring_attention_update - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现。reproducibility: pseudocode。
# 表达设计意图：Ring Attention 输出合并（在线 softmax rescale）。
# 来源：attention/ring_attention_update/ (d28d8f9...), README L18-28.

import triton
import triton.language as tl


@triton.jit
def ring_attention_update_kernel(
    prev_attn_out_ptr, prev_softmax_max_ptr, prev_softmax_sum_ptr,
    cur_attn_out_ptr, cur_softmax_max_ptr, cur_softmax_sum_ptr,
    attn_out_ptr, softmax_max_out_ptr, softmax_sum_out_ptr,
    n_elements,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # 读 prev
    prev_out = tl.load(prev_attn_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    prev_max = tl.load(prev_softmax_max_ptr + offs, mask=mask, other=0.0)
    prev_sum = tl.load(prev_softmax_sum_ptr + offs, mask=mask, other=0.0)

    # 读 cur
    cur_out = tl.load(cur_attn_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cur_max = tl.load(cur_softmax_max_ptr + offs, mask=mask, other=0.0)
    cur_sum = tl.load(cur_softmax_sum_ptr + offs, mask=mask, other=0.0)

    # 在线 softmax 合并
    new_max = tl.maximum(prev_max, cur_max)
    alpha_prev = tl.exp(prev_max - new_max)
    alpha_cur = tl.exp(cur_max - new_max)
    new_sum = prev_sum * alpha_prev + alpha_cur
    attn_out = prev_out * alpha_prev / new_sum + cur_out * alpha_cur / new_sum

    # 写回
    tl.store(attn_out_ptr + offs, attn_out, mask=mask)
    tl.store(softmax_max_out_ptr + offs, new_max, mask=mask)
    tl.store(softmax_sum_out_ptr + offs, new_sum, mask=mask)


# grid = (n_elements // BLOCK,)
# SBH layout: offs = b*S*H + s*H + h (batch, seq, head)
# TND layout: offs = t*H + h (cumulative token, head)
# 这是纯 elementwise 算子（无 Cube/AIC 参与），AIV-only 语义。
# CANN 6 tiling key 分支（0/1/2=SBH+FP16/BF16/FLOAT, 10/11/12=TND+同上）
#   在 Triton 中由 dtype constexpr + layout constexpr 分支处理。