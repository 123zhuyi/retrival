# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import typing

import numpy as np
import torch

__spec__ = {
    "bias": "BiasKernelSpec",
}


_TOL = {
    "float32": {"standard": "stat_rel_err", "level": "L1"},
    "float16": {"standard": "stat_rel_err", "level": "L1"},
    "bfloat16": {"standard": "stat_rel_err", "level": "L1"},
}


def _attr(kwargs, name, default):
    v = kwargs.get(name, default)
    if isinstance(v, str):
        s = v.strip()
        if isinstance(default, bool):
            sl = s.lower()
            if sl in ("true", "false", "yes", "no", "1", "0"):
                return sl in ("true", "yes", "1")
            return default
        try:
            return type(default)(s)
        except (TypeError, ValueError):
            return default
    return v


def _to_torch_dtype(dtype_str):
    dtype_str = str(dtype_str)
    if "bfloat16" in dtype_str:
        return torch.bfloat16
    if "float16" in dtype_str:
        return torch.float16
    return torch.float32


def _bias_reshape_shape(x, bias, axis, num_axes, bias_from_blob):
    rank = x.ndim
    if axis < 0:
        axis += rank
    if bias_from_blob:
        if num_axes == -1:
            return [1] * axis + list(bias.shape)
        if num_axes == 0:
            return [1] * rank
        return [1] * axis + list(bias.shape) + [1] * (rank - axis - num_axes)
    if bias.ndim == 1 and bias.shape[0] == 1:
        return [1] * rank
    return [1] * axis + list(bias.shape) + [1] * (rank - axis - bias.ndim)


def _compute(*tensors, **kwargs):
    """全程 torch.Tensor 进、torch.Tensor 出。返回 list[Tensor]。

    R3: golden 使用 torch.add，非 numpy 纯公式。
    """
    x, bias = tensors[0], tensors[1]
    axis = _attr(kwargs, "axis", 1)
    num_axes = _attr(kwargs, "num_axes", 1)
    bias_from_blob = _attr(kwargs, "bias_from_blob", True)
    reshape_shape = _bias_reshape_shape(x, bias, axis, num_axes, bias_from_blob)
    bias_reshaped = bias.reshape(reshape_shape)
    return [torch.add(x, bias_reshaped)]


class _Compose:
    """torch 原生算子拼出等价语义。"""

    def __init__(self, **kwargs):
        self.axis = _attr(kwargs, "axis", 1)
        self.num_axes = _attr(kwargs, "num_axes", 1)
        self.bias_from_blob = _attr(kwargs, "bias_from_blob", True)

    def __call__(self, x, bias, **kwargs):
        """Explicit inputs keep remote name-based binding from treating attrs as tensors."""
        reshape_shape = _bias_reshape_shape(
            x, bias, self.axis, self.num_axes, self.bias_from_blob
        )
        bias_reshaped = bias.reshape(reshape_shape)
        return [torch.add(x, bias_reshaped)]


class BiasKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。"""

    def golden(*inputs, **kwargs):
        t = []
        for arr in inputs:
            if isinstance(arr, np.ndarray):
                if arr.dtype.kind == "f" and arr.dtype != np.float64:
                    arr = arr.astype(np.float32)
                elif arr.dtype.kind != "f":
                    arr = arr.astype(np.float32)
                arr = np.ascontiguousarray(arr)
                t.append(torch.from_numpy(arr))
        outs = _compute(*t, **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        _NP_DTYPE = {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.float32,
        }
        result = []
        for i, o in enumerate(outs):
            if i < len(od):
                target_dtype = _to_torch_dtype(od[i])
                o = o.to(target_dtype)
            np_arr = o.to(torch.float32).cpu().numpy()
            np_dt = _NP_DTYPE.get(od[i] if i < len(od) else "float32", np.float32)
            result.append(np_arr.astype(np_dt))
        return result

    third_party: typing.ClassVar[dict] = {"torch": _Compose}
    tolerance: typing.ClassVar[dict] = _TOL
