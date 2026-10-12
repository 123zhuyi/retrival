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

"""Stable Sort references for the Kernel/GEIR and ACLNN pathways.

PyTorch ``torch.sort`` is the native competitor.  All computation is done by
that API; array/tensor conversion below only adapts the Kernel/GEIR boundary.
The repository does not deliver an ONNX pathway.  TTK does not execute ONNX.
"""

import torch

__spec__ = {
    "sort": "SortKernelSpec",
    "aclnnSort": "SortAclnnSpec",
    "torch.sort": "SortTorchSpec",
}

__golden__ = {
    "kernel": {"sort": "sort_golden"},
    "aclnn": {"aclnnSort": "aclnn_sort_golden"},
}
__input__ = {"kernel": {"sort": "sort_input"}}

_TOLERANCE = {
    dtype: {
        "standard": "stat_rel_err"
        if dtype in ("float16", "float32", "bfloat16")
        else "binary_equal"
    }
    for dtype in (
        "float16",
        "float32",
        "bfloat16",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    )
}


def _from_array(x):
    """Convert the Kernel/GEIR array without changing its value bits."""
    if "bfloat16" in str(x.dtype):
        return torch.from_numpy(x.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(x)


def _to_array(x):
    x = x.detach().cpu().contiguous()
    if x.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return x.view(torch.int16).numpy().view(bfloat16)
    return x.numpy()


def _require_stable(stable):
    if not bool(stable):
        raise ValueError("Sort ST covers the stable=True contract only")


def aclnn_sort_golden(
    self, stable=0, dim=0, descending=0, valuesOut=None, indicesOut=None, **kwargs
):
    """Golden reference compatible with the ACLNN asset registry."""
    return torch.sort(input=self, dim=dim, descending=descending, stable=True)


def _index_dtype(y2_dtype=3, output_dtypes=()):
    if output_dtypes and len(output_dtypes) > 1:
        dtype = output_dtypes[1]
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        return torch.int64 if "int64" in str(dtype) else torch.int32
    return torch.int64 if int(y2_dtype) == 9 else torch.int32


def _sort(x, axis, descending, stable, index_dtype):
    _require_stable(stable)
    values, indices = torch.sort(
        x, dim=int(axis), descending=bool(descending), stable=True
    )
    if indices.dtype != index_dtype:
        indices = indices.to(index_dtype)
    return values, indices


def _fill_pattern_along_axis(tensor, values, axis):
    """Repeat a pattern inside each logical row of the sorted axis."""
    axis = int(axis)
    if axis < 0:
        axis += tensor.ndim
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"axis {axis} is out of bounds for rank {tensor.ndim}")
    pattern = torch.tensor(values, dtype=tensor.dtype)
    repeats = (tensor.numel() + pattern.numel() - 1) // pattern.numel()
    moved_shape = [tensor.shape[i] for i in range(tensor.ndim) if i != axis]
    moved_shape.append(tensor.shape[axis])
    data = pattern.repeat(repeats)[: tensor.numel()].reshape(moved_shape)
    tensor.copy_(data.movedim(-1, axis))


def sort_input(x, *, axis=-1, testcase_name="", **kwargs):
    """Inject deterministic tie, signed-zero, NaN and infinity cases."""
    name = str(testcase_name)
    patterns = {
        "duplicate": [2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0],
        "signed_zero": [-0.0, 0.0, 0.0, -0.0, 1.0, -1.0, 0.0, -0.0],
        "nan": [3.0, float("nan"), 1.0, float("nan"), 2.0, 2.0, -1.0, 0.0],
        "infinity": [float("inf"), 3.0, -float("inf"), 0.0, 3.0, -0.0, 1.0, -1.0],
    }
    for marker, values in patterns.items():
        if name.endswith(f"_{marker}"):
            tensor = _from_array(x)
            _fill_pattern_along_axis(tensor, values, axis)
            break
    return (x,)


def sort_golden(
    x, *, axis=-1, descending=False, stable=True, y2_dtype=3, output_dtypes=(), **kwargs
):
    values, indices = _sort(
        _from_array(x),
        axis,
        descending,
        stable,
        _index_dtype(y2_dtype, output_dtypes),
    )
    return [_to_array(values), _to_array(indices)]


class SortThirdParty:
    """Timed competitor: ``__call__`` contains only required Sort work."""

    def __init__(
        self,
        *,
        axis=-1,
        descending=False,
        stable=True,
        y2_dtype=3,
        output_dtypes=(),
        **kwargs,
    ):
        _require_stable(stable)
        self.axis = int(axis)
        self.descending = bool(descending)

    def __call__(self, x, **kwargs):
        return torch.sort(x, dim=self.axis, descending=self.descending, stable=True)


class SortKernelSpec:
    golden = staticmethod(sort_golden)
    customize_inputs = staticmethod(sort_input)
    third_party = {"torch": SortThirdParty}
    tolerance = dict(_TOLERANCE)


class SortAclnnThirdParty:
    def __init__(self, *, stable=True, dim=-1, descending=False, **kwargs):
        _require_stable(stable)
        self.dim = int(dim)
        self.descending = bool(descending)

    def __call__(self, *args, **kwargs):
        return torch.sort(
            args[0], dim=self.dim, descending=self.descending, stable=True
        )


def _aclnn_sort_golden(x, stable=True, dim=-1, descending=False, *_outputs, **kwargs):
    _require_stable(stable)
    return list(aclnn_sort_golden(x, stable, int(dim), bool(descending)))


class SortAclnnSpec:
    golden = staticmethod(_aclnn_sort_golden)
    third_party = {"torch": SortAclnnThirdParty}
    tolerance = dict(_TOLERANCE)


class SortTorchSpec(SortAclnnSpec):
    """torch/torch_npu delivery path; argument semantics match ``torch.sort``."""
