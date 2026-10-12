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

import functools

import numpy as np

__golden__ = {"kernel": {"ascend_quant": "ascend_quant_golden"}}


def ascend_quant_golden(
    x, *, scale, offset, sqrt_mode=False, round_mode="Round", dst_type=2, **kwargs
):
    """
    Golden function for ascend_quant.
    All the parameters (names and order) follow @ascend_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    input_x = x
    x_shape = input_x.shape
    dtype = input_x.dtype
    out_format = kwargs["output_formats"][0]

    short_soc_version = kwargs.get("short_soc_version")

    if short_soc_version in ["Ascend950"]:
        cast_fp32 = input_x.astype("float32")
        scale = np.float32(scale)
        offset = np.float32(offset)

        if sqrt_mode:
            scale_rst = cast_fp32 * (scale * scale)
        else:
            scale_rst = cast_fp32 * scale
        add_offset = scale_rst + offset

        if round_mode == "Round":
            round_data = (
                np.round(add_offset, 8)
                if str(kwargs["output_dtypes"][0])
                in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
                else np.rint(add_offset)
            )
        elif round_mode == "Floor":
            round_data = np.floor(add_offset)
        elif round_mode == "Ceil":
            round_data = np.ceil(add_offset)
        else:
            round_data = np.trunc(add_offset)

        if dst_type == 2:
            round_data = np.clip(round_data, -128, 127)
            output = round_data.astype("int8", copy=False)
        elif dst_type == 29:
            from ml_dtypes import int4

            round_data = np.clip(round_data, -8, 7)
            output = round_data.astype(int4, copy=False)
        elif dst_type == 34:
            from en_dtypes import hifloat8

            output = round_data.astype(hifloat8, copy=False)
        elif dst_type == 35:
            from ml_dtypes import float8_e5m2

            np.clip(round_data, a_min=-57344.0, a_max=57344.0, out=round_data)
            round_data = np.nan_to_num(round_data, nan=0.0, copy=False)
            output = round_data.astype(float8_e5m2, copy=False)
        elif dst_type == 36:
            from ml_dtypes import float8_e4m3fn

            output = round_data.astype(float8_e4m3fn, copy=False)

    else:
        if out_format in ("ND", "NCHW"):
            cast_fp16 = input_x.astype("float16")
            scale = np.float32(scale).astype("float16")
            offset = np.float32(offset).astype("float16")
            if sqrt_mode:
                scale_sqrt = cast_fp16 * scale
                scale_rst = scale_sqrt * scale
            else:
                scale_rst = cast_fp16 * scale
            add_offset = scale_rst + offset
            if round_mode == "Round":
                round_data = np.rint(add_offset)
            elif round_mode == "Floor":
                round_data = np.floor(add_offset)
            elif round_mode == "Ceil":
                round_data = np.ceil(add_offset)
            else:
                round_data = np.trunc(add_offset)
            if dst_type == 2:
                round_data = np.clip(round_data, -128, 127)
                output = round_data.astype("int8")
            else:
                from ml_dtypes import int4

                output = round_data.astype(int4)
        else:
            if out_format == "NC1HWC0":
                N, C1, H, W, C0 = input_x.shape
            elif out_format == "NDC1HWC0":
                N_pre, D, C1, H, W, C0 = input_x.shape
                N = N_pre * D
            elif out_format == "FRACTAL_NZ":
                N = 1
                if len(x_shape) > 4:
                    N = functools.reduce(lambda x, y: x * y, x_shape[:-4])
                C1, H, W, C0 = [x_shape[-4], x_shape[-3], x_shape[-2], x_shape[-1]]
            out_c1 = (C1 + 1) // 2
            out_shape = (N, out_c1, H, W, 32)
            output = np.zeros(shape=out_shape, dtype="int8")
            x_shape_new = (N, (C1 + 1) // 2 * 2, H, W, 16)
            input_x_new = np.zeros(shape=x_shape_new, dtype=dtype)

            for c1 in range(C1):
                if out_format == "NC1HWC0":
                    input_x_new[:, c1, :, :, :] = input_x[:, c1, :, :, :]
                elif N == 1:
                    input_x_new[0, c1, :, :, :] = input_x[c1, :, :, :]

            if dtype == "float32":
                cast_fp16 = input_x_new.astype("float16")
            else:
                cast_fp16 = input_x_new

            mul_scale = cast_fp16 * scale
            if sqrt_mode:
                mul_scale = mul_scale * scale

            add_offset = mul_scale + offset

            if round_mode == "Round":
                round_data = np.rint(add_offset)
            elif round_mode == "Floor":
                round_data = np.floor(add_offset)
            elif round_mode == "Ceil":
                round_data = np.ceil(add_offset)
            else:
                round_data = np.trunc(add_offset)

            round_data[round_data < -128] = -128
            round_data[round_data > 127] = 127

            out = round_data.astype("int8")

            out_c0 = 32
            for c1_idx in range(out_shape[1]):
                for c0_idx in range(out_c0):
                    src_c1 = (c1_idx * out_c0 + c0_idx) // 16
                    src_c0 = (c1_idx * out_c0 + c0_idx) % 16
                    output[:, c1_idx, :, :, c0_idx] = out[:, src_c1, :, :, src_c0]
            if out_format == "NDC1HWC0":
                output = output.reshape([N_pre, D, C1, H, W, C0])

    return output
