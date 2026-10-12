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

import numpy as np
import torch
import hashlib

__spec__ = {
    "cumulative_logsumexp": "CumulativeLogsumexpKernelSpec",
}

__golden__ = {
    "kernel": {
        "cumulative_logsumexp": "cumulative_logsumexp_golden",
    }
}

__input__ = {
    "kernel": {
        "cumulative_logsumexp": "cumulative_logsumexp_input",
    }
}


def _stable_rng(testcase_name):
    seed = int(hashlib.md5(str(testcase_name).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _range_for_x(input_ranges):
    try:
        low, high = input_ranges[0]
    except (TypeError, IndexError):
        return -8.0, 8.0
    if low is None or high is None:
        return -8.0, 8.0
    return float(low), float(high)


def _attr(kwargs, name, default):
    value = kwargs.get(name)
    if value is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            value = attrs.get(name)
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "false", "yes", "no", "1", "0"):
            return lowered in ("true", "yes", "1")
    return value


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def cumulative_logsumexp_input(
    x, axis, *, testcase_name="", input_ranges=None, **kwargs
):
    rng = _stable_rng(testcase_name)
    low, high = _range_for_x(input_ranges)
    x_data = rng.uniform(low, high, x.shape).astype(x.dtype, copy=False)
    return x_data, axis


def _normalize_axis(axis, rank):
    axis = (
        int(axis.item())
        if hasattr(axis, "item")
        else int(np.asarray(axis).reshape(-1)[0])
    )
    if axis < 0:
        axis += rank
    return axis


def _compute_torch_logcumsumexp(tensor, axis, exclusive=False, reverse=False):
    if reverse:
        tensor = torch.flip(tensor, dims=(axis,))
    result = torch.logcumsumexp(tensor, dim=axis)
    if exclusive:
        first_shape = list(result.shape)
        first_shape[axis] = 1
        first = torch.full(first_shape, -float("inf"), dtype=result.dtype)
        tail = result.narrow(axis, 0, result.shape[axis] - 1)
        result = torch.cat((first, tail), dim=axis)
    if reverse:
        result = torch.flip(result, dims=(axis,))
    return result


def _torch_logcumsumexp(x, axis, exclusive, reverse):
    tensor = torch.from_numpy(np.ascontiguousarray(x))
    result = _compute_torch_logcumsumexp(tensor, axis, exclusive, reverse)
    return result.numpy()


def _compute(x, axis, **kwargs):
    exclusive = bool(_attr(kwargs, "exclusive", False))
    reverse = bool(_attr(kwargs, "reverse", False))
    normalized_axis = _normalize_axis(axis, x.dim())
    return [_compute_torch_logcumsumexp(x, normalized_axis, exclusive, reverse)]


def cumulative_logsumexp_golden(x, axis, *, exclusive=False, reverse=False, **kwargs):
    axis = _normalize_axis(axis, x.ndim)
    out_dtype = x.dtype
    result = _torch_logcumsumexp(x, axis, exclusive, reverse)
    return result.astype(out_dtype, copy=False)


class _CumulativeLogsumexpCompose:
    """Third-party reference executed on the remote GPU server."""

    def __init__(self, exclusive=False, reverse=False, **kwargs):
        self.exclusive = bool(
            _attr({"exclusive": exclusive, **kwargs}, "exclusive", False)
        )
        self.reverse = bool(_attr({"reverse": reverse, **kwargs}, "reverse", False))

    def __call__(self, x, axis, **kwargs):
        out = _compute_torch_logcumsumexp(
            x,
            _normalize_axis(axis, x.dim()),
            self.exclusive,
            self.reverse,
        )
        return [out.to(dtype=x.dtype)]


class CumulativeLogsumexpKernelSpec:
    """kernel + geir shared spec. The golden entry receives and returns numpy arrays."""

    def golden(x, axis, **kwargs):
        tensor = torch.from_numpy(np.ascontiguousarray(x))
        outs = _compute(tensor, axis, **kwargs)
        output_dtype = _output_dtype(kwargs, 0, str(np.asarray(x).dtype))
        return [outs[0].cpu().numpy().astype(output_dtype, copy=False)]

    third_party = {"torch": _CumulativeLogsumexpCompose}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }


# 【不存在】aclnn 通路: math/cumulative_logsumexp/CMakeLists.txt 使用
# ACLNNTYPE aclnn_exclude, 不暴露公开 aclnn API.
# 【不存在】e2e 通路: 未新增 torch_npu eager 绑定.
