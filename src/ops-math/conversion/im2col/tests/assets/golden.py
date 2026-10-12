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
    "kernel": {
        "im2col": "im2col_golden"
    }
}


def im2col_golden(x, ksizes, strides=[1], dilations=[1], padding_mode="CALCULATED", pads=[0], **kwargs):
    '''
    Kernel golden for im2col.
    All the parameters follow @im2col_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    x_format = kwargs.get('input_ori_formats', ['NCHW'])[0]
    y_format = kwargs.get('output_formats', ['NCHW'])[0]
    y_shape = kwargs.get('output_ori_shapes', None)
    
    if y_shape is None:
        y_shape = kwargs.get('output_shapes', None)
    
    # Get NCHW dimensions from input
    if x_format == "NHWC":
        N, C, H, W = x.shape[0], x.shape[3], x.shape[1], x.shape[2]
        x = x.transpose(0, 3, 1, 2)
    else:
        N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    
    # Get output dimensions (y_shape is a tuple containing the shape tuple)
    if x_format == "NHWC":
        Co, Ho, Wo = y_shape[0][3], y_shape[0][1], y_shape[0][2]
    else:
        Co, Ho, Wo = y_shape[0][1], y_shape[0][2], y_shape[0][3]
    
    # Normalize single-element lists
    if len(strides) == 1:
        strides = [strides[0], strides[0]]
    if len(dilations) == 1:
        dilations = [dilations[0], dilations[0]]
    if len(pads) == 1:
        pads = [pads[0], pads[0], pads[0], pads[0]]
    
    # Handle integer dtypes via view/cast
    cast = False
    view = False
    xdtype = x.dtype
    if x.dtype in [np.int8, np.uint8, np.bool_]:
        x = x.astype(np.int16)
        x = x.view(np.float16)
        cast = True
    elif x.dtype in [np.int16, np.uint16]:
        x = x.view(np.float16)
        view = True
    elif x.dtype in [np.int32, np.uint32]:
        x = x.view(np.float32)
        view = True
    elif x.dtype in [np.int64, np.uint64, np.complex64]:
        x = x.view(np.float64)
        view = True
    
    # Determine padding based on mode
    if padding_mode == "SAME":
        pads_ = _calc_need_padding(H, W, ksizes, strides, dilations)
    elif padding_mode == "VALID":
        pads_ = [0, 0, 0, 0]
    else:
        # CALCULATED: pads = [pad_h_before, pad_w_before, pad_h_after, pad_w_after]
        pads_ = pads
    
    # Apply asymmetric padding: np.pad handles the difference, torch.unfold handles the symmetric part
    x_pad = np.pad(x, [(0, 0), (0, 0), 
                       (max(0, pads_[0] - pads_[1]), max(0, pads_[1] - pads_[0])),
                       (max(0, pads_[2] - pads_[3]), max(0, pads_[3] - pads_[2]))], "constant")
    pads_symmetric = [min(pads_[0], pads_[1]), min(pads_[2], pads_[3])]
    
    x_tensor = torch.from_numpy(x_pad)
    x_unfold = torch.nn.functional.unfold(x_tensor, ksizes, dilations, pads_symmetric, strides)
    x_unfold = x_unfold.numpy()
    
    # Convert back to original dtype
    if cast:
        x_unfold = x_unfold.view(np.int16)
        x_unfold = x_unfold.astype(xdtype)
    elif view:
        x_unfold = x_unfold.view(xdtype)
    
    # Reshape to output dimensions
    res = x_unfold.reshape([N, Co, Ho, Wo])
    
    # Convert to output format if needed
    if y_format == "NHWC":
        res = res.transpose(0, 2, 3, 1)
    
    return res


def _calc_need_padding(H, W, ksizes, strides, dilations):
    """Calculate padding needed for SAME mode."""
    effectH = (ksizes[0] - 1) * dilations[0] + 1
    effectW = (ksizes[1] - 1) * dilations[1] + 1
    outputHSize = H // strides[0] if (H % strides[0] == 0) else H // strides[0] + 1
    outputWSize = W // strides[1] if (W % strides[1] == 0) else W // strides[1] + 1

    hNeedPadding = (outputHSize - 1) * strides[0] + effectH - H
    if hNeedPadding < 0:
        hNeedPadding = 0
    wNeedPadding = (outputWSize - 1) * strides[1] + effectW - W
    if wNeedPadding < 0:
        wNeedPadding = 0

    hPaddingBefore = hNeedPadding // 2
    hPaddingAfter = hNeedPadding - hPaddingBefore
    wPaddingBefore = wNeedPadding // 2
    wPaddingAfter = wNeedPadding - wPaddingBefore
    return [hPaddingBefore, wPaddingBefore, hPaddingAfter, wPaddingAfter]
