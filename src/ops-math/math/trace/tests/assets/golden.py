#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

__golden__ = {
    "kernel": {"trace": "trace_golden"},
    "aclnn": {"aclnnTrace": "aclnn_trace_golden"},
}


def trace_golden(x, **kwargs):
    """
    Golden function for trace.
    All the parameters (names and order) follow @trace_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    x_dtype = x.dtype
    output_dtype = None
    declared_output_dtypes = kwargs.get("output_dtypes")
    if declared_output_dtypes:
        declared_output_dtype = declared_output_dtypes[0]
        try:
            output_dtype = np.dtype(declared_output_dtype)
        except (TypeError, ValueError):
            # Some environments expose bfloat16 through an extended numpy dtype.
            # In that case x.dtype is already the dtype required by the OpDef.
            output_dtype = None

    # torch.trace() on CPU does not support float16/bfloat16/uint16/uint32/bool.
    # Strategy: compute in a compatible dtype, then convert result back.
    if x_dtype.name in ("bfloat16", "float16"):
        # bf16/fp16: accumulate in float32, then return the declared output dtype.
        x_fp32 = x.astype(np.float32)
        x_torch = torch.from_numpy(x_fp32)
        result = torch.trace(x_torch)
        target_dtype = output_dtype if output_dtype is not None else x_dtype
        output = result.numpy().astype(target_dtype, copy=False)
    elif x_dtype.name in ("uint16", "uint32", "bool"):
        # uint16/uint32/bool: convert to int64 for computation
        x_i64 = x.astype(np.int64)
        x_torch = torch.from_numpy(x_i64)
        result = torch.trace(x_torch)
        output = result.numpy()
    else:
        # float32, float64, int8, int16, int32, int64, uint8, complex64, complex128
        x_torch = torch.from_numpy(x)
        result = torch.trace(x_torch)
        output = result.numpy()

    if output_dtype is not None and output.dtype != output_dtype:
        output = output.astype(output_dtype, copy=False)

    return output


def _aclnn_trace_impl(self_tensor):
    """Common implementation for aclnnTrace.

    torch.trace() on CPU does not support float16/bfloat16/uint16/uint32/bool.
    Compute in a compatible dtype instead.
    """
    if self_tensor.dtype in (torch.float16, torch.bfloat16):
        # Accumulate in float32. aclnn_trace_golden casts to out.dtype below.
        return torch.trace(self_tensor.to(torch.float32))
    elif self_tensor.dtype == torch.bool:
        return torch.trace(self_tensor.to(torch.int64))
    else:
        # Check for uint16/uint32 (torch doesn't have native uint16/uint32,
        # but if passed as int64 from numpy conversion, it's already fine)
        return torch.trace(self_tensor)


def aclnn_trace_golden(selfT, out, **kwargs):
    """
    Aclnn golden for aclnnTrace.
    All the parameters (name & order) follow \
        function `aclnnTraceGetWorkspaceSize` in @aclnn_trace.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    if isinstance(selfT, np.ndarray):
        self_dtype = selfT.dtype
        if self_dtype.name in ("bfloat16", "float16"):
            # Compute in float32 (numpy has no bfloat16; TTK handles bf16 conversion)
            self_fp32 = selfT.astype(np.float32)
            self_torch = torch.from_numpy(self_fp32)
        elif self_dtype.name in ("uint16", "uint32", "bool"):
            # Convert to int64 for computation
            self_i64 = selfT.astype(np.int64)
            self_torch = torch.from_numpy(self_i64)
        else:
            self_torch = torch.from_numpy(selfT)
    else:
        self_torch = selfT

    result = _aclnn_trace_impl(self_torch)

    if isinstance(out, np.ndarray):
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        return np.asarray(result).astype(out.dtype, copy=False)

    if isinstance(result, np.ndarray):
        result = torch.from_numpy(result)
    return result.to(dtype=out.dtype)
