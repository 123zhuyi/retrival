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
"""Segsum multi pathway golden.

Pathways registered here match 01_requirement.md section 3.3:
    kernel  supported, op_kernel/arch35 implementation
    aclnn   supported, aclnnExpSegsum
    geir    not supported, no infershape is registered for Segsum
    e2e     not supported, torch_npu binary never references aclnnExpSegsum
"""

import hashlib
import numpy as np
import torch

__spec__ = {
    "segsum": "SegsumKernelSpec",
    "aclnnExpSegsum": "SegsumAclnnSpec",
}

__golden__ = {
    "kernel": {
        "segsum": "segsum_golden",
    }
}

__input__ = {
    "kernel": {
        "segsum": "segsum_input",
    }
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


def _inject_non_finite(arr):
    """Sprinkle +inf / -inf / nan into a copy of arr at fixed positions.

    DFX cases assert the propagation contract (02_design 6.1): non-finite values
    flow through the fp32 accumulation and exp exactly as torch does, and the
    strictly-upper triangle stays 0 regardless of the input. Positions are fixed
    rather than random so a rerun compares against the very same golden.
    """
    flat = arr.reshape(-1)
    if flat.size == 0:
        return arr
    specials = [np.inf, -np.inf, np.nan]
    for i, value in enumerate(specials):
        if i < flat.size:
            flat[i] = value
    if flat.size > 8:  # 再往中段放一组,覆盖非首块的传播
        for i, value in enumerate(specials):
            flat[flat.size // 2 + i] = value
    return arr


def segsum_input(x, *, testcase_name="", input_ranges=None, **kwargs):
    """Deterministic input so a rerun of the same case compares against the same golden."""
    rng = _stable_rng(testcase_name)
    low, high = _range_at(input_ranges, 0)
    data = rng.uniform(low, high, x.shape).astype(x.dtype, copy=False)
    if "infnan" in str(testcase_name):
        data = _inject_non_finite(data)
    return (data,)


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


def _compute(x_t):
    """out[..., i, j] = exp(sum of x[..., j + 1 : i + 1]) for j <= i, 0 otherwise.

    Built from torch primitives only (expand, tril mask, cumsum, masked fill, exp),
    the same decomposition the interface document describes. Half and bfloat16
    accumulate in float32, mirroring the kernel, and are cast back at the end.
    """
    out_dtype = x_t.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x_t = x_t.to(torch.float32)
    t_dim = x_t.shape[-1]

    # broadcast the last dim into a (T, T) block: expanded[..., i, j] = x[..., i]
    expanded = x_t.unsqueeze(-1).expand(*x_t.shape, t_dim)
    zero = torch.zeros((), dtype=torch.float32, device=x_t.device)
    strict_tril = torch.tril(
        torch.ones(t_dim, t_dim, dtype=torch.bool, device=x_t.device), diagonal=-1
    )
    masked = torch.where(strict_tril, expanded, zero)
    seg = torch.cumsum(masked, dim=-2, dtype=torch.float32)
    tril = torch.tril(
        torch.ones(t_dim, t_dim, dtype=torch.bool, device=x_t.device), diagonal=0
    )
    neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=x_t.device)
    seg = torch.where(tril, seg, neg_inf)
    y = torch.exp(seg)
    if out_dtype in (torch.float16, torch.bfloat16):
        y = y.to(out_dtype)
    return [y]


def segsum_golden(x, **kwargs):
    out_dtype = getattr(x, "dtype", None)
    x_t = torch.as_tensor(_to_f32_array(x), dtype=torch.float32)
    y = _compute(x_t)[0]

    result = y.cpu().numpy()
    out_dtype_name = (
        getattr(out_dtype, "name", str(out_dtype)).lower()
        if out_dtype is not None
        else ""
    )
    if "float16" in out_dtype_name or "bfloat16" in out_dtype_name:
        return result.astype(_numpy_dtype(out_dtype), copy=False)
    return result.astype(np.float32, copy=False)


class _SegsumCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, x=None, *tensors, self_=None, **kwargs):
        if x is None:
            x = self_ if self_ is not None else tensors[0]
        return _compute(x)


class SegsumKernelSpec:
    """kernel spec. The golden entry receives numpy arrays."""

    def golden(x, **kwargs):
        x_t = torch.as_tensor(_to_f32_array(x), dtype=torch.float32)
        outs = _compute(x_t)
        output_dtype = _output_dtype(kwargs, 0, str(np.asarray(x).dtype))
        return [outs[0].cpu().numpy().astype(_numpy_dtype(output_dtype), copy=False)]

    third_party = {"torch": _SegsumCompose}


class SegsumAclnnSpec:
    """aclnnExpSegsum spec. The golden entry receives torch tensors.

    The parameter name follows aclnn_segsum.h, where the input is named self.
    """

    def golden(self, **kwargs):
        return _compute(self)

    third_party = {"torch": _SegsumCompose}


# 【不存在】geir 通路: 全仓无 IMPL_OP_INFERSHAPE(Segsum), op_graph 下无 proto/infer 实现.
# 【不存在】e2e 通路: strings libtorch_npu.so | grep aclnnExpSegsum 命中 0.
