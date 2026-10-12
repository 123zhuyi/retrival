# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================

"""SignBitsPack TestSpec — golden + third_party + tolerance.

按 TestSpec 规范编写，覆盖 Kernel/GEIR 和 ACLNN 通路。

通路注册：
  - Kernel/GEIR: sign_bits_pack → SignBitsPackTestSpec
  - ACLNN:       aclnnSignBitsPack → AclnnSignBitsPackTestSpec

精度升精度策略：
  - golden: FP16/BF16 输入在计算前统一升精度到 FP32（计算前全部完成，不在计算过程中升精度）
  - third_party: 不升精度，按原始 dtype 计算（性能对标用）

竞品情况：SignBitsPack 是 CANN 特有算子，TF/PyTorch 无原生竞品。
按规范"无竞品算子，通过 torch/tf 小算子拼接"，golden 与 third_party 均用
torch 小算子组合实现（符号位提取 + MSB-first 位打包）。

精度标准：输出 uint8，bit-exact（binary_equal），rtol=0 / atol=0。
"""

import torch

__spec__ = {
    "sign_bits_pack": "SignBitsPackTestSpec",
    "aclnnSignBitsPack": "AclnnSignBitsPackTestSpec",
}

_MSB_WEIGHTS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)

_UPCAST_DTYPES = (torch.float16, torch.bfloat16)


def _sign_bits_pack_torch(x: torch.Tensor, size: int) -> torch.Tensor:
    """SignBitsPack 核心计算（torch 小算子拼接，无竞品接口）。

    MSB-first: bit7=第1个元素符号位, bit0=第8个。
    b_i = (x_i < 0): 严格小于，+0/-0/+Inf → 0, 负数/-Inf/-1.0 → 1。
    尾部非8倍数时用 -1.0 填充（符号位=1）。
    """
    x_flat = x.reshape(-1)
    n = x_flat.shape[0]
    pad = (-n) % 8
    if pad:
        x_flat = torch.cat([x_flat, torch.full((pad,), -1.0, dtype=x_flat.dtype)])
    bits = (x_flat < 0).to(torch.uint8)
    bits = bits.reshape(-1, 8)
    packed = (bits * _MSB_WEIGHTS.to(bits.dtype)).sum(dim=1).to(torch.uint8)
    packed_len = (n + 7) // 8
    return packed.reshape(size, packed_len // size)


class SignBitsPackTestSpec:
    """Kernel / GEIR 通路 — golden 收 numpy.ndarray，third_party 收 torch.Tensor。"""

    def golden(x, *, size=1, **kwargs):
        x_t = torch.from_numpy(x)
        # FP16/BF16 升精度到 FP32（计算前全部完成，不在计算过程中升精度）
        if x_t.dtype in _UPCAST_DTYPES:
            x_t = x_t.to(torch.float32)
        return [_sign_bits_pack_torch(x_t, int(size)).numpy()]

    class ThirdPartyImpl:
        """三方标杆 — 不升精度，按原始 dtype 计算。"""

        def __init__(self, *, size=1, **kwargs):
            self.size = int(size)

        def __call__(self, x, **kwargs):
            return [_sign_bits_pack_torch(x, self.size)]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
    }


class AclnnSignBitsPackTestSpec:
    """ACLNN 通路 — golden / third_party 均收 torch.Tensor。"""

    def golden(selfT, size, out=None, **kwargs):
        # FP16/BF16 升精度到 FP32（计算前全部完成，不在计算过程中升精度）
        if selfT.dtype in _UPCAST_DTYPES:
            selfT = selfT.to(torch.float32)
        result = _sign_bits_pack_torch(selfT, int(size))
        if out is not None:
            out.copy_(result)
            return [out]
        return [result]

    class ThirdPartyImpl:
        """三方标杆 — 不升精度，按原始 dtype 计算。"""

        def __init__(self, size, out=None, **kwargs):
            self.size = int(size)

        def __call__(self, selfT, out=None, **kwargs):
            return [_sign_bits_pack_torch(selfT, self.size)]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
    }
