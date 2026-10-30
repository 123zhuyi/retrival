# moe_finalize_routing_v2_grad - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：MoE 路由最终化反向（scatter-add + 可选 scale 梯度）
#   3 种场景（无 scale / 有 scale / 有 scale+bias）
#   expandedRowIdx 驱动的 scatter-add
# CANN 专有机制（MTE DataCopyPad / KERNEL_TYPE_MIX_AIV_1_0）不在此体现。
# 来源：moe/moe_finalize_routing_v2_grad/ (commit d28d8f9...), README.md.

import triton
import triton.language as tl


@triton.jit
def moe_finalize_routing_v2_grad_kernel(
    grad_y_ptr,                     # [R, H]
    expanded_row_idx_ptr,           # [R * K]
    expanded_x_ptr,                 # [R * K, H] 或 [activeNum, H]
    scales_ptr,                     # [R, K] 可选
    expert_idx_ptr,                 # [R, K] 可选
    bias_ptr,                       # [E, H] 可选
    grad_expanded_x_ptr,            # [R * K, H] 或 [activeNum, H] 输出
    grad_scales_ptr,                # [R * K] 输出
    R, K, H,
    HAS_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CUT_H: tl.constexpr,            # 沿 H 维切分
):
    if CUT_H:
        pid_h = tl.program_id(0)    # 沿 H 维切分
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    else:
        offs_h = tl.arange(0, H)

    for i in range(R * K):
        idx = tl.load(expanded_row_idx_ptr + i)  # expandedRowIdx[i]
        h_mask = offs_h < H

        # gradY[i/K, :]
        gy = tl.load(grad_y_ptr + (i // K) * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)

        if HAS_SCALE:
            s = tl.load(scales_ptr + i).to(tl.float32)  # scales[i]
            gy = gy * s

        # gradExpandedXOut[expandedRowIdx[i]] += gy  (scatter-add)
        tl.atomic_add(grad_expanded_x_ptr + idx * H + offs_h, gy, mask=h_mask)

        if HAS_SCALE:
            if HAS_BIAS:
                exp = tl.load(expanded_x_ptr + idx * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
                ex = tl.load(expert_idx_ptr + i).to(tl.int32)
                b = tl.load(bias_ptr + ex * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
                grad_scales = tl.sum((exp + b) * gy, axis=0)
            else:
                exp = tl.load(expanded_x_ptr + idx * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
                grad_scales = tl.sum(exp * gy, axis=0)
            tl.store(grad_scales_ptr + i, grad_scales)


# grid = (ceil(H / BLOCK_H),) if CUT_H else (1,)
# 纯 Vector 计算（scatter-add + elemwise + reduce），无 Cube