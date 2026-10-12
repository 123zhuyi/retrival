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

__golden__ = {
    "kernel": {"one_hot": "one_hot_golden"},
    "aclnn": {"aclnnOneHot": "aclnn_one_hot_golden"},
}


def one_hot_golden(x, depth, on_value, off_value, axis: int = -1, **kwargs):
    """
    Kernel golden for one_hot.
    All the parameters follow @one_hot_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
              input_formats, output_formats, input_ori_formats, output_ori_formats,
              input_dtypes, output_dtypes.
    """
    import tensorflow.compat.v1 as tf
    from tensorflow.python.ops import gen_array_ops

    tf.disable_eager_execution()

    data_dtype = on_value.dtype
    on_value_const = tf.constant(on_value, shape=(), dtype=data_dtype)
    off_value_const = tf.constant(off_value, shape=(), dtype=data_dtype)
    axis = max(axis, -1)
    out = gen_array_ops.one_hot(
        x, max(int(depth), 0), on_value_const, off_value_const, axis
    )
    with tf.Session() as sess:
        res = sess.run(out)
    return res


def _to_torch_tensor(arr):
    """Convert numpy.ndarray (incl. bfloat16) to torch.Tensor."""
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr
    np_dtype = arr.dtype.name
    if "bfloat16" in np_dtype:
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _scalar_value(t):
    """Extract Python scalar from 0-dim / 1-elem tensor or ndarray."""
    if isinstance(t, (torch.Tensor,)):
        return t.item() if t.numel() == 1 else t.flatten()[0].item()
    if isinstance(t, np.ndarray):
        return t.item() if t.size == 1 else t.flatten()[0].item()
    return t


def aclnn_one_hot_golden(self, numClasses, onValue, offValue, axis, out, **kwargs):
    """
    Aclnn golden for aclnnOneHot.
    Params (name & order) follow aclnnOneHotGetWorkspaceSize in @aclnn_one_hot.h
    without workspaceSize & executor:
        (self, numClasses, onValue, offValue, axis, out)
    All params are passed positionally by the framework (including output `out`).
    - self/onValue/offValue/out: tensors (torch.Tensor or numpy.ndarray).
    - numClasses/axis: C int/int64_t values.
    Returns: output tensor matching NPU aclnnOneHot semantics.
    """
    idx = _to_torch_tensor(self)
    on_v = _scalar_value(
        onValue if isinstance(onValue, (torch.Tensor, np.ndarray)) else onValue
    )
    off_v = _scalar_value(
        offValue if isinstance(offValue, (torch.Tensor, np.ndarray)) else offValue
    )

    # Infer output dtype from out tensor (authoritative), fallback to onValue.
    if out is not None:
        out_t = _to_torch_tensor(out)
        out_dtype = out_t.dtype if out_t is not None else torch.get_default_dtype()
    elif isinstance(onValue, (torch.Tensor, np.ndarray)):
        out_dtype = _to_torch_tensor(onValue).dtype
    else:
        out_dtype = torch.get_default_dtype()

    depth = max(int(numClasses), 0)
    idx_long = idx.long()
    # NPU kernel treats out-of-range indices (< 0 or >= depth) as off_value.
    # torch.nn.functional.one_hot raises on out-of-range, so mark invalid
    # indices with a sentinel (depth, which is out-of-range for one_hot) and
    # use a mask to set those positions to off_value afterwards.
    invalid_mask = (idx_long < 0) | (idx_long >= depth)
    idx_safe = idx_long.clamp(0, max(depth - 1, 0))
    oh = torch.nn.functional.one_hot(idx_safe, depth)  # [..., depth], int64
    result = torch.where(
        oh.bool(),
        torch.tensor(on_v, dtype=out_dtype),
        torch.tensor(off_v, dtype=out_dtype),
    )
    # Overwrite positions where the original index was out-of-range with off_value.
    if invalid_mask.any():
        result[invalid_mask] = off_v
    # one_hot appends new dim at -1; move to the specified axis.
    result = torch.movedim(result, -1, int(axis))
    return result
