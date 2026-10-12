#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
import tensorflow as tf


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def im2col_nchw(input_x,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w):
    N, C, H, W = input_x.shape

    out_h = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    col = np.zeros((N, C * kernel_h * kernel_w, out_h * out_w),
                   dtype=input_x.dtype)

    for n in range(N):
        col_idx = 0
        for oh in range(out_h):
            for ow in range(out_w):
                row_idx = 0
                for c in range(C):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            ih = oh * stride_h + kh * dilation_h - pad_h
                            iw = ow * stride_w + kw * dilation_w - pad_w
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                col[n, row_idx, col_idx] = input_x[n, c, ih, iw]
                            else:
                                col[n, row_idx, col_idx] = 0
                            row_idx += 1
                col_idx += 1
    return col


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype
    }

    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)

    if len(shape) != 4:
        print("Im2Col only supports NCHW 4D input.")
        exit(1)

    # 生成输入
    size = np.prod(shape)
    tmp_input = np.random.uniform(-1.0, 1.0, size=size)
    tmp_input = tmp_input.reshape(shape).astype(np_type)

    # Im2Col 固定参数（典型 3x3 same conv）
    kernel_h = 3
    kernel_w = 3
    stride_h = 1
    stride_w = 1
    pad_h = 1
    pad_w = 1
    dilation_h = 1
    dilation_w = 1

    tmp_golden = im2col_nchw(
        tmp_input,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    )

    tmp_input.tofile(f"{d_type}_input_im2col.bin")
    tmp_golden.tofile(f"{d_type}_golden_im2col.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)

    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
