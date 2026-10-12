#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Bit-exact kernel/GEIR golden for TensorRedirect in TTK TestSpec format."""

import numpy as np
import torch


__spec__ = {
    # Kernel and GEIR share the snake-case registration and the same TestSpec.
    "tensor_redirect": "TensorRedirectKernelSpec",
}

# Retain the repository-facing legacy entry while consumers migrate to TestSpec.
__golden__ = {
    "kernel": {"tensor_redirect": "tensor_redirect_golden"},
}


_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
}


def _numpy_dtype(dtype):
    """Resolve TTK dtype values, including NumPy's optional bfloat16 dtype."""
    name = getattr(dtype, "name", str(dtype)).lower()
    if name in ("bf16", "bfloat16"):
        try:
            from ml_dtypes import bfloat16
        except ImportError as exc:
            raise RuntimeError(
                "TensorRedirect bfloat16 golden requires the optional ml-dtypes package"
            ) from exc
        return bfloat16
    return np.dtype(dtype)


def _output_dtypes(kwargs):
    values = kwargs.get("output_dtypes") or ()
    return [
        value[0] if isinstance(value, (list, tuple)) and value else value
        for value in values
    ]


def _to_torch_bit_exact(array):
    """Convert NumPy to torch without changing any payload bits."""
    array = np.asarray(array)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _to_numpy_bit_exact(tensor):
    """Convert torch to NumPy while preserving BF16 payload bits."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(_numpy_dtype("bfloat16"))
    return tensor.numpy()


def _compute(x):
    """Copy through the independent PyTorch reference interface."""
    return [torch.clone(x)]


def _kernel_golden(x, **kwargs):
    outputs = [
        _to_numpy_bit_exact(output) for output in _compute(_to_torch_bit_exact(x))
    ]
    output_dtypes = _output_dtypes(kwargs)
    return [
        output.astype(_numpy_dtype(output_dtypes[index]), copy=False)
        if index < len(output_dtypes)
        else output
        for index, output in enumerate(outputs)
    ]


class TensorRedirectKernelSpec:
    """TestSpec shared by the TensorRedirect kernel and GEIR pathways."""

    @staticmethod
    def golden(x, **kwargs):
        return _kernel_golden(x, **kwargs)

    third_party = {"torch": "torch.clone"}
    tolerance = _TOLERANCE


def tensor_redirect_golden(x, **kwargs):
    """Legacy kernel entry backed by the same TestSpec computation."""
    if not kwargs.get("output_dtypes"):
        kwargs = {**kwargs, "output_dtypes": (x.dtype,)}
    return _kernel_golden(x, **kwargs)[0]


# 【不存在】ACLNN 通路：CMakeLists.txt 显式配置 ACLNNTYPE aclnn_exclude。
# 【不存在】e2e 通路：本算子不交付 ACLNN 符号，torch_npu 无对应绑定入口。
