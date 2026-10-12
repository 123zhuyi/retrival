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

__golden__ = {"kernel": {"gelu_quant": "gelu_quant_golden"}}


def _dynamic_quant_common(
    x,
    smooth_scales=None,
    group_index=None,
    dst_type=2,
    quant_mode="pertoken",
    is_symmetrical=True,
    output_dtype=None,
):
    scale_max = np.float32(0.0)
    scale_max_no_sym = np.float32(0.0)
    if dst_type == 2:
        scale_max = np.float32(127.0)
        scale_max_no_sym = np.float32(255.0)
    elif dst_type == 29:
        scale_max = np.float32(7.0)
        scale_max_no_sym = np.float32(15.0)
    elif dst_type == 34:
        scale_max = np.float32(32768.0)
        scale_max_no_sym = np.float32(65536.0)
    elif dst_type == 35:
        scale_max = np.float32(57344.0)
        scale_max_no_sym = np.float32(114688.0)
    elif dst_type == 36:
        scale_max = np.float32(448.0)
        scale_max_no_sym = np.float32(896.0)

    if smooth_scales is not None:
        smooth_scales = smooth_scales.astype("float32")
    else:
        smooth_scales = 1

    if group_index is not None:
        x = x.reshape(-1, x.shape[-1])
        S, H = x.shape
        E = group_index.shape[0]
        input_mul = np.empty(shape=(0, H), dtype="float32")
        for row_scale in range(E):
            x_start_row = 0 if row_scale == 0 else group_index[row_scale - 1]
            x_end_row = group_index[row_scale]
            if x_start_row < x_end_row:
                mul_rows = x[x_start_row:x_end_row] * smooth_scales[row_scale]
                input_mul = np.concatenate([input_mul, mul_rows], axis=0)
    else:
        x = x.astype("float32")
        input_mul = x * smooth_scales

    offset = None
    if is_symmetrical is False:
        input_abs = input_mul
        input_max = (
            np.max(input_abs)
            if quant_mode == "pertensor"
            else np.max(input_abs, axis=-1, keepdims=True)
        )
        input_min = (
            np.min(input_abs)
            if quant_mode == "pertensor"
            else np.min(input_abs, axis=-1, keepdims=True)
        )
        scale = (input_max - input_min) * (np.float32(1.0) / scale_max_no_sym)
        offset = scale_max - (input_max / scale)
        input_scaled = input_mul / scale + offset
    else:
        input_abs = np.abs(input_mul)
        input_max = (
            np.max(input_abs)
            if quant_mode == "pertensor"
            else np.max(input_abs, axis=-1, keepdims=True)
        )
        scale = input_max * (np.float32(1.0) / scale_max)
        input_scaled = input_mul / scale

    round_data = (
        input_scaled
        if str(output_dtype) in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
        else np.round(input_scaled, 0)
    )

    if dst_type == 2:
        round_data = round_data.astype("int8", copy=False)
    elif dst_type == 29:
        from ml_dtypes import int4

        round_data = round_data.astype(int4, copy=False)
    elif dst_type == 35:
        from ml_dtypes import float8_e5m2

        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        from ml_dtypes import float8_e4m3fn

        round_data = round_data.astype(float8_e4m3fn, copy=False)
    elif dst_type == 34:
        from ml_dtypes import hifloat8

        round_data = round_data.astype(hifloat8, copy=False)

    if is_symmetrical is False:
        output_data = [round_data, scale.squeeze(-1), offset.squeeze(-1)]
    else:
        output_data = [round_data, scale.squeeze(-1)]
    return output_data


def gelu_quant_golden(
    x,
    input_scale=None,
    input_offset=None,
    *,
    approximate="none",
    quant_mode="dynamic",
    dst_type=2,
    round_mode="rint",
    **kwargs,
):
    """
    Golden function for gelu_quant.
    All the parameters (names and order) follow @gelu_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    if x.dtype in ["float16", "bfloat16"]:
        x = x.astype(np.float32)
    x_torch = torch.from_numpy(x)
    gelu_res = torch.nn.functional.gelu(x_torch, approximate=approximate)
    y = gelu_res.numpy()
    if quant_mode == "static":
        scale = input_scale
        offset = input_offset

        cast_dtype = "float32"
        scale = scale.astype(cast_dtype)
        offset = offset.astype(cast_dtype) if offset is not None else None

        scale_rst = y * scale
        if offset is not None:
            scale_rst = scale_rst + offset

        round_data = (
            scale_rst
            if str(kwargs["output_dtypes"][0])
            in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
            else np.round(scale_rst, 0)
        )

        if dst_type == 2:
            round_data = np.clip(round_data, -128, 127)
            round_data = round_data.astype("int8", copy=False)
        elif dst_type == 35:
            from ml_dtypes import float8_e5m2

            round_data = round_data.astype(float8_e5m2, copy=False)
        elif dst_type == 36:
            from ml_dtypes import float8_e4m3fn

            round_data = round_data.astype(float8_e4m3fn, copy=False)
        elif dst_type == 34:
            from ml_dtypes import hifloat8

            round_data = round_data.astype(hifloat8, copy=False)

        return round_data, None
    else:
        return _dynamic_quant_common(
            y,
            smooth_scales=input_scale,
            dst_type=dst_type,
            output_dtype=kwargs["output_dtypes"][0],
        )
