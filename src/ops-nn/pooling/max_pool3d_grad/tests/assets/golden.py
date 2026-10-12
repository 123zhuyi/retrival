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

__golden__ = {"kernel": {"max_pool3d_grad": "max_pool3d_grad_golden"}}


def max_pool3d_grad_golden(
    orig_x,
    orig_y,
    grads,
    *,
    ksize,
    strides,
    padding="SAME",
    pads=None,
    data_format="NDHWC",
    **kwargs,
):
    """
    Golden function for max_pool3d_grad.
    All the parameters (names and order) follow @max_pool3d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    input_dtype = orig_x.dtype
    if input_dtype.name == "bfloat16":
        orig_x_t = (
            torch.from_numpy(orig_x.view(np.int16))
            .view(torch.bfloat16)
            .to(torch.float32)
        )
        orig_y_t = (
            torch.from_numpy(orig_y.view(np.int16))
            .view(torch.bfloat16)
            .to(torch.float32)
        )
        grads_t = (
            torch.from_numpy(grads.view(np.int16))
            .view(torch.bfloat16)
            .to(torch.float32)
        )
    else:
        orig_x_t = torch.from_numpy(orig_x).to(torch.float32)
        orig_y_t = torch.from_numpy(orig_y).to(torch.float32)
        grads_t = torch.from_numpy(grads).to(torch.float32)

    x_shape = orig_x.shape
    y_shape = orig_y.shape

    if data_format == "NDHWC":
        n, d, h, w, c = x_shape
        yn, yd, yh, yw, yc = y_shape
        orig_x_t = orig_x_t.permute(0, 4, 1, 2, 3).contiguous()
        orig_y_t = orig_y_t.permute(0, 4, 1, 2, 3).contiguous()
        grads_t = grads_t.permute(0, 4, 1, 2, 3).contiguous()
    elif data_format == "NCDHW":
        n, c, d, h, w = x_shape
        yn, yc, yd, yh, yw = y_shape
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")

    kd, kh, kw = ksize[0], ksize[1], ksize[2]
    sd, sh, sw = strides[0], strides[1], strides[2]

    if padding == "SAME":
        pad_d = max((yd - 1) * sd + kd - d, 0)
        pad_h = max((yh - 1) * sh + kh - h, 0)
        pad_w = max((yw - 1) * sw + kw - w, 0)
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padding_tuple = (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right)
    elif padding == "VALID":
        padding_tuple = (0, 0, 0, 0, 0, 0)
    else:
        if pads is not None and len(pads) == 6:
            padding_tuple = (pads[0], pads[1], pads[2], pads[3], pads[4], pads[5])
        elif pads is not None and len(pads) == 3:
            padding_tuple = (pads[0], pads[0], pads[1], pads[1], pads[2], pads[2])
        else:
            padding_tuple = (0, 0, 0, 0, 0, 0)

    orig_x_padded = torch.nn.functional.pad(orig_x_t, padding_tuple)

    grad_output = torch.zeros_like(orig_x_padded)
    flat_grads = grads_t.reshape(n, c, -1)
    flat_orig_y = orig_y_t.reshape(n, c, -1)

    for nn in range(n):
        for cc in range(c):
            for pd in range(yd):
                for ph in range(yh):
                    for pw in range(yw):
                        d_start = pd * sd
                        h_start = ph * sh
                        w_start = pw * sw
                        d_end = min(
                            d_start + kd, d + padding_tuple[0] + padding_tuple[1]
                        )
                        h_end = min(
                            h_start + kh, h + padding_tuple[2] + padding_tuple[3]
                        )
                        w_end = min(
                            w_start + kw, w + padding_tuple[4] + padding_tuple[5]
                        )

                        window = orig_x_padded[
                            nn, cc, d_start:d_end, h_start:h_end, w_start:w_end
                        ]
                        y_val = flat_orig_y[nn, cc, pd * yh * yw + ph * yw + pw]
                        grad_val = flat_grads[nn, cc, pd * yh * yw + ph * yw + pw]

                        mask = window == y_val
                        if mask.sum() > 0:
                            grad_window = grad_output[
                                nn, cc, d_start:d_end, h_start:h_end, w_start:w_end
                            ]
                            grad_window[mask] = grad_val

    pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding_tuple
    if (
        pad_front > 0
        or pad_back > 0
        or pad_top > 0
        or pad_bottom > 0
        or pad_left > 0
        or pad_right > 0
    ):
        grad_output = grad_output[
            :,
            :,
            pad_front : pad_front + d,
            pad_top : pad_top + h,
            pad_left : pad_left + w,
        ]

    if data_format == "NDHWC":
        grad_output = grad_output.permute(0, 2, 3, 4, 1).contiguous()

    result = grad_output.numpy()
    if input_dtype.name == "bfloat16":
        return result.astype(np.float32).view(np.int16).astype(input_dtype, copy=False)
    elif input_dtype == np.float16:
        return result.astype(np.float16, copy=False)
    else:
        return result.astype(input_dtype, copy=False)
