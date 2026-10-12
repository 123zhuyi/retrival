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


__golden__ = {"kernel": {"matrix_set_diag_v2": "matrix_set_diag_v2_golden"}}


def matrix_set_diag_v2_golden(x, diagonal, k, **kwargs):
    """
    Kernel golden for matrix_set_diag_v2.
    All the parameters (names and order) follow @matrix_set_diag_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: 待替换的原始张量 (numpy.ndarray)，ND，rank ∈ [2, 8]，最后两维为矩阵 (rowNum × colNum)。
            dtype 支持 BOOL/COMPLEX64/DOUBLE/FLOAT/BF16/FLOAT16/INT8~INT64/UINT8~UINT64。
        diagonal: 对角线张量 (numpy.ndarray)，dtype 与 x 一致。
            k0 == k1 (单对角线) 时 rank = x.rank − 1，最后一维为单条对角线 (长度 = maxDiagLen)；
            k0 != k1 (多对角线) 时 rank = x.rank，最后两维为 [numDiags, maxDiagLen]，
            其中 numDiags = k1 − k0 + 1，且 diagonal[..., j, :] 对应偏移量 k1 − j
            (j=0 → k1 最高对角线，j=numDiags−1 → k0 最低对角线)。
        k: 对角线偏移 (numpy.ndarray)，INT32。标量或长度为 2 的向量。
            k[0] = k0 (下界，更负)，k[1] = k1 (上界，更正)，保证 k0 <= k1；标量时 k0 = k1。

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor: 与 x 同 shape/dtype，将 x 的指定对角线元素替换为 diagonal 的对应值。
    """
    y = x.copy()

    k_arr = np.asarray(k).reshape(-1).astype(np.int64)
    if k_arr.size == 1:
        k0 = k1 = int(k_arr[0])
    else:
        k0 = int(k_arr[0])
        k1 = int(k_arr[1])

    rowNum = x.shape[-2]
    colNum = x.shape[-1]
    single = k0 == k1

    for kk in range(k0, k1 + 1):
        row_start = max(0, -kk)
        row_end = min(rowNum, colNum - kk)
        if row_end <= row_start:
            continue
        rows = np.arange(row_start, row_end)
        cols = rows + kk
        elem_idx = rows if kk >= 0 else cols
        if single:
            diag_vals = diagonal[..., elem_idx]
        else:
            diag_vals = diagonal[..., k1 - kk, elem_idx]
        y[..., rows, cols] = diag_vals

    return y
