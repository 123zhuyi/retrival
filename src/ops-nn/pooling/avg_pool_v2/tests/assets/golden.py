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


__golden__ = {"kernel": {"avg_pool_v2": "avg_pool_v2_golden"}}


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def _getPadList(
    padding,
    x_shape,
    w_shape,
    dy_shape,
    strides,
    dilations=(1, 1, 1, 1),
    pads=None,
    ceil_mode=False,
):
    N, C, H, W = x_shape
    Co, _, hk, wk = w_shape
    strideh, stridew = strides
    _, _, dilationh, dilationw = dilations
    He = (hk - 1) * dilationh + 1
    We = (wk - 1) * dilationw + 1
    if padding == "VALID":
        pad = [0, 0, 0, 0]
        if dy_shape is None:
            Ho = (H - He) // strideh + 1
            Wo = (W - We) // stridew + 1
        else:
            N, Co, Ho, Wo = dy_shape
    elif padding == "SAME":
        if dy_shape is None:
            Ho = (H + strideh - 1) // strideh
            Wo = (W + stridew - 1) // stridew
        else:
            N, Co, Ho, Wo = dy_shape
        padh = max(0, (Ho - 1) * strideh + He - H)
        padw = max(0, (Wo - 1) * stridew + We - W)
        pad = [padh // 2, padh // 2 + padh % 2, padw // 2, padw // 2 + padw % 2]
    elif padding == "CALCULATED":
        if pads is None:
            raise RuntimeError("pads cannot be None when padding=CALCULATED")
        pad = pads
        if dy_shape is None:
            padt, padb, padl, padr = pads
            if ceil_mode:
                Ho = (H - He + padt + padb + strideh - 1) // strideh + 1
                Wo = (W - We + padl + padr + stridew - 1) // stridew + 1
            else:
                Ho = (H - He + padt + padb) // strideh + 1
                Wo = (W - We + padl + padr) // stridew + 1
        else:
            N, Co, Ho, Wo = dy_shape
    else:
        raise RuntimeError("not support this mode:{} yet".format(padding))

    return pad, [N, Co, Ho, Wo]


def avg_pool_v2_golden(
    x,
    ksize,
    strides,
    padding_mode="CALCULATED",
    pads=(0, 0, 0, 0),
    data_format="NCHW",
    global_pooling=False,
    ceil_mode=False,
    exclusive=True,
    divisor_override=0,
    **kwargs,
):
    """
    Kernel golden for avg_pool_v2.
    All the parameters follow @avg_pool_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    padding = padding_mode
    output_dtype = kwargs.get("output_dtypes")
    # 5HD input only
    print("[WARNING] dynamic_avgpool golden func")
    if len(x.shape) != 5:
        raise RuntimeError(
            "avgpool testcase golden function supports NC1HWC0 input only!"
        )
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    ksize_h, ksize_w = ksize[h_index], ksize[w_index]
    IN, IC, IH, IW, C0 = x.shape
    # x filter to NCHW
    x = x.transpose(0, 1, 4, 2, 3).reshape(IN, IC * C0, IH, IW).astype(np.float32)

    # resize window
    if ksize_h > IH:
        ksize_h = IH
    if ksize_w > IW:
        ksize_w = IW
    if global_pooling or (ksize_h >= IH and ksize_w >= IW):
        ksize_h = IH
        ksize_w = IW
        padding = "VALID"
    pad, dy_shape = _getPadList(
        padding,
        [IN, IC * C0, IH, IW],
        [IC * C0, 16, ksize_h, ksize_w],
        None,
        [strideh, stridew],
        pads=pads,
        ceil_mode=ceil_mode,
    )
    padt, padb, padl, padr = pad
    _, _, OH, OW = dy_shape

    res_out_NCHW = np.zeros(dy_shape, dtype=np.float32)
    if padding == "VALID":
        valid_h = (OH - 1) * strideh + ksize_h
        valid_w = (OW - 1) * stridew + ksize_w
        valid_shape = (IN, IC * C0, valid_h, valid_w)
        tensor_in_with_pad = np.zeros(valid_shape, dtype=np.float16)
        tensor_in_with_pad[:, :, :valid_h, :valid_w] = x[:, :, :valid_h, :valid_w]
        for i in range(OH):
            for j in range(OW):
                tensor_in_mask_with_window = tensor_in_with_pad[
                    :,
                    :,
                    i * strideh : i * strideh + ksize_h,
                    j * stridew : j * stridew + ksize_w,
                ]
                res_out_NCHW[:, :, i, j] = (
                    1.0
                    * np.sum(tensor_in_mask_with_window, axis=(2, 3))
                    / (ksize_h * ksize_w)
                )
    else:
        same_shape = (IN, IC * C0, IH + padt + padb, IW + padl + padr)
        tensor_in_with_pad = np.zeros(same_shape, dtype=np.float16)
        tensor_in_with_pad[:, :, padt : IH + padt, padl : IW + padl] = x
        # compute avg sum
        for i in range(OH):
            for j in range(OW):
                tensor_in_mask_with_window = tensor_in_with_pad[
                    :,
                    :,
                    i * strideh : i * strideh + ksize_h,
                    j * stridew : j * stridew + ksize_w,
                ]
                res_out_NCHW[:, :, i, j] = np.sum(
                    tensor_in_mask_with_window, axis=(2, 3)
                )
        # compute mean factor
        avg_mean_factor = []
        for i in range(OH):
            for j in range(OW):
                h_start = i * strideh - padt
                w_start = j * stridew - padl
                h_end = min(h_start + ksize_h, IH)
                w_end = min(w_start + ksize_w, IW)
                h_start = max(h_start, 0)
                w_start = max(w_start, 0)

                area = max((h_end - h_start) * (w_end - w_start), 1)
                mean_value = 1.0 / float(area)
                avg_mean_factor.append(mean_value)
        avg_mean_factor = np.array(avg_mean_factor).reshape(OH, OW)
        # compute res out
        for i in range(OH):
            for j in range(OW):
                res_out_NCHW[:, :, i, j] = (
                    res_out_NCHW[:, :, i, j] * avg_mean_factor[i, j]
                )

    # NCHW to NC1HWC0
    output = res_out_NCHW.reshape((IN, IC, C0, OH, OW)).transpose((0, 1, 3, 4, 2))
    if output_dtype[0] == "float16":
        output = due_fp16_overflow(output.astype(np.float16))
    return output
