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

__golden__ = {
    "kernel": {"tabulate_fusion": "tabulate_fusion_golden"},
}


def tabulate_fusion_golden(table, table_info, em_x, em, *, last_layer_size, **kwargs):
    """
    Golden function for tabulate_fusion.
    All the parameters (names and order) follow SE doc prototype definition without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        table: np.ndarray, shape=(N, last_size_align*6), dtype=float32/float16
            查表数据，每行6组系数 a0~a5，每组 last_size_align 个元素。
            last_size_align = ceil(last_layer_size / 64) * 64
        table_info: np.ndarray, shape=(>=5,), dtype=float32/float16
            查表参数，前5个元素为 [lower, upper, max, stride0, stride1]
        em_x: np.ndarray, shape=(nloc*nnei, 1), dtype=float32/float16
            环境嵌入值，每个 nloc 有 nnei 个 xx 值
        em: np.ndarray, shape=(nloc, nnei, 4), dtype=float32/float16
            环境矩阵，em[nloc_i, nnei_j, 0:4] 为 4 个 ll 系数
        last_layer_size: int
            输出最后一维大小

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor: descriptor, shape=(nloc, 4, last_layer_size), dtype=em.dtype
    """
    # 确保属性为 Python int
    last_layer_size = int(last_layer_size)

    # 提取查表参数
    lower = float(table_info[0])
    upper = float(table_info[1])
    max_val = float(table_info[2])
    stride0 = float(table_info[3])
    stride1 = float(table_info[4])

    # 预计算 stride
    first_stride = int(np.floor((upper - lower) / stride0))
    second_stride = int(np.floor((max_val - upper) / stride1))

    # 对齐大小
    last_size_align = ((last_layer_size + 63) // 64) * 64

    # 获取维度信息
    nloc = em.shape[0]
    nnei = em.shape[1]

    # 确保 em_x 是一维展开
    em_x_flat = em_x.reshape(-1)  # shape=(nloc*nnei,)

    # 初始化输出
    out_dtype = em.dtype
    descriptor = np.zeros((nloc, 4, last_layer_size), dtype=out_dtype)

    # 遍历每个 nloc
    for nloc_i in range(nloc):
        res = np.zeros((4, last_layer_size), dtype=np.float64)  # 高精度累加

        for nnei_j in range(nnei):
            xx = float(em_x_flat[nloc_i * nnei + nnei_j])

            # 分支定位：计算 table_idx 和 xx_new
            if lower <= xx < upper:
                # 分支2
                table_idx = int(np.floor((xx - lower) / stride0))
                xx_new = xx - (table_idx * stride0 + lower)
            elif upper <= xx < max_val:
                # 分支3
                table_idx = first_stride + int(np.floor((xx - upper) / stride1))
                xx_new = xx - ((table_idx - first_stride) * stride1 + upper)
            elif xx >= max_val:
                # 分支4
                table_idx = first_stride + second_stride - 1
                xx_new = 0.0
            else:
                # xx < lower（边界保护，原TBE未显式处理）
                table_idx = 0
                xx_new = 0.0

            # 边界保护：table_idx 不能越界
            if table_idx < 0:
                table_idx = 0
            if table_idx >= table.shape[0]:
                table_idx = table.shape[0] - 1

            # 从 table 读取 a0~a5（每组 last_size_align 个元素，取前 last_layer_size 个）
            # 显式提升为 float64 进行高精度中间计算（对齐 SE 文档 §6/§7.2 要求）
            # 避免 numpy 1.26 值基提升规则导致 float16/float32 退化为低精度
            row = table[table_idx]  # shape=(last_size_align*6,)
            a0 = row[
                0 * last_size_align : 0 * last_size_align + last_layer_size
            ].astype(np.float64)
            a1 = row[
                1 * last_size_align : 1 * last_size_align + last_layer_size
            ].astype(np.float64)
            a2 = row[
                2 * last_size_align : 2 * last_size_align + last_layer_size
            ].astype(np.float64)
            a3 = row[
                3 * last_size_align : 3 * last_size_align + last_layer_size
            ].astype(np.float64)
            a4 = row[
                4 * last_size_align : 4 * last_size_align + last_layer_size
            ].astype(np.float64)
            a5 = row[
                5 * last_size_align : 5 * last_size_align + last_layer_size
            ].astype(np.float64)

            # xx_new 为 Python float（float64 精度），参与 float64 数组运算时自动提升
            # 5次多项式 (Horner法则): var = a0 + (a1 + (a2 + (a3 + (a4 + a5*xx_new)*xx_new)*xx_new)*xx_new)*xx_new)*xx_new
            # 中间计算全程 float64（对齐 SE 文档 §6 "float64 高精度中间累加"）
            var = (
                a0
                + (a1 + (a2 + (a3 + (a4 + a5 * xx_new) * xx_new) * xx_new) * xx_new)
                * xx_new
            )  # shape=(last_layer_size,), dtype=float64

            # 读取 ll 系数（Python float，float64 精度，参与 float64 数组运算时自动提升）
            ll_0 = float(em[nloc_i, nnei_j, 0])
            ll_1 = float(em[nloc_i, nnei_j, 1])
            ll_2 = float(em[nloc_i, nnei_j, 2])
            ll_3 = float(em[nloc_i, nnei_j, 3])

            # 计算并累加 out = var * [ll_0, ll_1, ll_2, ll_3]
            # var 为 float64，ll 为 Python float，乘法结果为 float64；res 为 float64 高精度累加器
            res[0] += var * ll_0
            res[1] += var * ll_1
            res[2] += var * ll_2
            res[3] += var * ll_3

        descriptor[nloc_i] = res.astype(out_dtype)

    return descriptor
