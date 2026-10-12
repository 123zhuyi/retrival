#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__golden__ = {"kernel": {"trilu": "trilu_golden"}}


def trilu_golden(x, k=None, *, upper=0, **kwargs):
    """Golden function for Trilu."""

    if upper not in (0, 1):
        raise ValueError(f"upper must be 0 or 1, but got {upper}")

    if k is None:
        diagonal = 0
    else:
        k_array = np.asarray(k)
        if k_array.size != 1:
            raise ValueError(
                f"k must contain exactly one element, but got shape {k_array.shape}"
            )
        diagonal = int(k_array.item())

    input_dtypes = kwargs.get("input_dtypes", [])
    is_complex32 = bool(input_dtypes) and input_dtypes[0] == "complex32"

    if is_complex32:
        # Complex32 在 Golden 框架中按 [..., M, N, 2] 存储：
        # 最后一维分别为 FP16 实部和 FP16 虚部。
        if x.ndim < 3 or x.shape[-1] != 2:
            raise ValueError(
                "complex32 input must use [..., M, N, 2] storage, "
                f"but got shape {x.shape}"
            )
        logical_shape = x.shape[:-1]
    else:
        logical_shape = x.shape

    if len(logical_shape) < 2:
        raise ValueError(
            f"x rank must be at least 2, but got logical shape {logical_shape}"
        )

    tri_func = np.triu if upper == 1 else np.tril

    if is_complex32:
        real = x[..., 0]
        imag = x[..., 1]

        real_result = tri_func(real, diagonal)
        imag_result = tri_func(imag, diagonal)

        return np.stack((real_result, imag_result), axis=-1)

    original_dtype = x.dtype

    # 避免部分 NumPy 环境对 BF16 运算支持不完整。
    if original_dtype.name == "bfloat16":
        result = tri_func(x.astype(np.float32), diagonal)
        return result.astype(original_dtype, copy=False)

    return tri_func(x, diagonal)
