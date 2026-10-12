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
import hashlib
import numpy as np
import torch

__spec__ = {
    "exp_segsum_grad": "ExpSegsumGradKernelSpec",
    "aclnnExpSegsumBackward": "ExpSegsumGradAclnnSpec",
}

__golden__ = {
    "kernel": {
        "exp_segsum_grad": "exp_segsum_grad_golden",
    }
}

__input__ = {
    "kernel": {
        "exp_segsum_grad": "exp_segsum_grad_input",
    }
}

_KERNEL_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_ACLNN_TOLERANCE = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
}


def _to_f32_array(arr):
    if hasattr(arr, "astype"):
        return arr.astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def _stable_rng(testcase_name):
    seed = int(hashlib.md5(str(testcase_name).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _range_at(input_ranges, index):
    try:
        low, high = input_ranges[index]
    except (TypeError, IndexError):
        return -2.0, 2.0
    if low is None or high is None:
        return -2.0, 2.0
    return float(low), float(high)


def _random_like(template, rng, value_range):
    low, high = value_range
    return rng.uniform(low, high, template.shape).astype(template.dtype, copy=False)


def exp_segsum_grad_input(
    grad_output, grad_self, *, testcase_name="", input_ranges=None, **kwargs
):
    rng = _stable_rng(testcase_name)
    grad_output_data = _random_like(grad_output, rng, _range_at(input_ranges, 0))
    grad_self_data = _random_like(grad_self, rng, _range_at(input_ranges, 1))
    return grad_output_data, grad_self_data


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def _numpy_dtype(dtype):
    dtype_name = str(dtype).lower()
    if "bfloat16" in dtype_name:
        try:
            from ml_dtypes import bfloat16

            return bfloat16
        except ImportError:
            try:
                import tensorflow

                return tensorflow.bfloat16.as_numpy_dtype
            except ImportError as exc:
                raise RuntimeError(
                    "ml_dtypes or tensorflow is needed to support bfloat16 dtype"
                ) from exc
    return np.dtype(dtype)


def _compute(grad_output_t, grad_self_t):
    out_dtype = grad_output_t.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        grad_output_t = grad_output_t.to(torch.float32)
        grad_self_t = grad_self_t.to(torch.float32)
    t_dim = grad_output_t.shape[-1]

    tril_mask = torch.tril(
        torch.ones(t_dim, t_dim, dtype=torch.bool, device=grad_output_t.device),
        diagonal=0,
    )
    zero = torch.zeros((), dtype=torch.float32, device=grad_output_t.device)
    y = torch.where(tril_mask, grad_output_t * grad_self_t, zero)
    y = torch.flip(y, dims=[-2])
    y = torch.cumsum(y, dim=-2, dtype=torch.float32)
    y = torch.flip(y, dims=[-2])
    strict_tril = torch.tril(
        torch.ones(t_dim, t_dim, dtype=torch.bool, device=grad_output_t.device),
        diagonal=-1,
    )
    y = torch.where(strict_tril, y, zero)
    grad_input = torch.sum(y, dim=-1, dtype=torch.float32)
    if out_dtype in (torch.float16, torch.bfloat16):
        grad_input = grad_input.to(out_dtype)
    return [grad_input]


def exp_segsum_grad_golden(grad_output, grad_self, **kwargs):
    out_dtype = getattr(grad_output, "dtype", None)
    grad_output_t = torch.as_tensor(_to_f32_array(grad_output), dtype=torch.float32)
    grad_self_t = torch.as_tensor(_to_f32_array(grad_self), dtype=torch.float32)
    grad_input = _compute(grad_output_t, grad_self_t)[0]

    result = grad_input.cpu().numpy()
    out_dtype_name = (
        getattr(out_dtype, "name", str(out_dtype)).lower()
        if out_dtype is not None
        else ""
    )
    if "float16" in out_dtype_name or "bfloat16" in out_dtype_name:
        return result.astype(_numpy_dtype(out_dtype), copy=False)
    return result.astype(np.float32, copy=False)


class _ExpSegsumGradCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(
        self,
        grad_output=None,
        grad_self=None,
        *tensors,
        gradOutput=None,
        gradSelf=None,
        **kwargs,
    ):
        if grad_output is None:
            grad_output = gradOutput if gradOutput is not None else tensors[0]
        if grad_self is None:
            grad_self = gradSelf if gradSelf is not None else tensors[1]
        return _compute(grad_output, grad_self)


class ExpSegsumGradKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    def golden(grad_output, grad_self, **kwargs):
        grad_output_t = torch.as_tensor(_to_f32_array(grad_output), dtype=torch.float32)
        grad_self_t = torch.as_tensor(_to_f32_array(grad_self), dtype=torch.float32)
        outs = _compute(grad_output_t, grad_self_t)
        output_dtype = _output_dtype(kwargs, 0, str(np.asarray(grad_output).dtype))
        return [outs[0].cpu().numpy().astype(_numpy_dtype(output_dtype), copy=False)]

    third_party = {"torch": _ExpSegsumGradCompose}
    tolerance = _KERNEL_TOLERANCE


class ExpSegsumGradAclnnSpec:
    """aclnnExpSegsumBackward spec. The golden entry receives torch tensors."""

    def golden(gradOutput, gradSelf, **kwargs):
        return _compute(gradOutput, gradSelf)

    third_party = {"torch": _ExpSegsumGradCompose}
    tolerance = _ACLNN_TOLERANCE


# 【不存在】e2e 通路: 未发现 torch_npu eager/aten 绑定到 aclnnExpSegsumBackward.
