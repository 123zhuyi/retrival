"""dropout_with_add_softmax Triton 骨架（P2 产出，生成用伪码）

语义链（组件合成，extraction.md §1.1）：
    t = x1 + alpha * self                       # AxpyV2，fp32 FMA
    s = softmax(t, dim=-1)                      # SoftmaxV2，fp32 内部 + 减行最大
    mask = (philox_uniform < keep_prob)         # GenMask 语义；Triton 自生成（bit-exact 不可达，mapping §3）
    result_softmax = s
    result_dropout = where(mask, s * (1/keep_prob), 0)   # DoMaskV3
    输出 mask = mask（0/1，unpacked uint8；布局待 device 实测裁决）

约束（PTA 文档）：input 4D (N,C,H,W)，W ∈ {128,256,384,512}，(N*C)%32==0，dim=-1。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dropout_with_add_softmax_kernel(
    self_ptr, x1_ptr,           # 输入：self, x1（同 shape，连续）
    sm_ptr,                     # 输出 1：result_softmax
    do_ptr,                     # 输出 2：result_dropout
    mask_ptr,                   # 输出 3：mask（uint8，0/1）
    alpha,                      # float32 标量
    inv_keep,                   # float32 标量 = 1.0 / keep_prob（wrapper 预算）
    keep_prob,                  # float32 标量（mask 判定阈值）
    seed,                       # uint64 随机种子（自生成 mask 用）
    NUM_ROWS,                   # N*C*H
    BLOCK_W: tl.constexpr,      # = W（128/256/384/512 全载单 tile）
):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    offs_w = tl.arange(0, BLOCK_W)

    for row in range(pid, NUM_ROWS, NUM_CORE):
        offs = row * BLOCK_W + offs_w                      # W=BLOCK_W 无需列 mask

        # ---- 组件 1：AxpyV2（fp32 FMA）----
        x1 = tl.load(x1_ptr + offs).to(tl.float32)
        sf = tl.load(self_ptr + offs).to(tl.float32)
        t = x1 + alpha * sf

        # ---- 组件 2：SoftmaxV2（fp32，减行最大）----
        m = tl.max(t, axis=0)
        e = tl.exp(t - m)
        s = tl.sum(e, axis=0)
        sm = e / s
        tl.store(sm_ptr + offs, sm.to(sm_ptr.dtype.element_ty))

        # ---- 组件 3a：GenMask 语义（自生成 philox；bit-exact 不可达）----
        r = tl.rand(seed, offs)                            # philox4x32-10, (0,1)
        keep = r < keep_prob

        # ---- 组件 3b：DoMaskV3（y = mask ? x/keep_prob : 0）----
        do = tl.where(keep, sm * inv_keep, 0.0)
        tl.store(do_ptr + offs, do.to(do_ptr.dtype.element_ty))
        tl.store(mask_ptr + offs, keep.to(tl.uint8))


def dropout_with_add_softmax(self, x1, alpha, prob, seed=0):
    """wrapper：prob 为丢弃概率（PTA 契约）；keep_prob = 1 - prob"""
    assert self.dim() == 4 and self.shape == x1.shape
    assert self.is_contiguous() and x1.is_contiguous()
    N, C, H, W = self.shape
    assert W in (128, 256, 384, 512) and (N * C) % 32 == 0
    assert 0.0 <= prob < 1.0

    keep_prob = 1.0 - prob
    # keep_prob==1（prob=0）：inv_keep=1.0，mask 全 1，kernel 内自然成立，无需特判
    inv_keep = 1.0 / keep_prob

    out_sm = torch.empty_like(self)
    out_do = torch.empty_like(self)
    out_mask = torch.empty(self.shape, dtype=torch.uint8, device=self.device)

    NUM_ROWS = N * C * H
    grid = (triton.runtime.driver.active.utils.get_device_properties(
        self.device.index)["multiprocessor_count"],)
    dropout_with_add_softmax_kernel[grid](
        self, x1, out_sm, out_do, out_mask,
        float(alpha), float(inv_keep), float(keep_prob), seed,
        NUM_ROWS, BLOCK_W=W,
    )
    return out_mask, out_sm, out_do
