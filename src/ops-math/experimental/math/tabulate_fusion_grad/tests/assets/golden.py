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

"""
Golden 实现文件 for tabulate_fusion_grad。

基于 SE 文档 §6 的 numpy 自定义实现（无竞品同名算子）。
TabulateFusionGrad 是 DeepMD-kit TabulateFusion 算子的反向梯度算子。
前向多项式: res(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
反向梯度:   grad(x) = a1 + 2*a2*x + 3*a3*x^2 + 4*a4*x^3 + 5*a5*x^4

dtype 支持: 仅 float32（SE §5.8 数据类型组合）。
"""

import numpy as np

__golden__ = {
    "kernel": {"tabulate_fusion_grad": "tabulate_fusion_grad_golden"},
}


def tabulate_fusion_grad_golden(
    table,
    table_info,
    em_x,
    em,
    dy,
    descriptor,
    *,
    split_count=1,
    split_index=0,
    **kwargs,
):
    """
    Golden function for tabulate_fusion_grad.
    All the parameters (names and order) follow SE doc prototype definition (REG_OP) without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        table: np.ndarray, shape=(N_table, 6*align64(L)), dtype=float32
            查找表，每行存 6 段系数 [a0, a1, a2, a3, a4, a5]，每段长度 align64(L)
        table_info: np.ndarray, shape=(6,), dtype=float32
            [lower, upper, max, stride0, stride1, rsv] 分段参数
        em_x: np.ndarray, shape=(nloc*nnei, 1), dtype=float32
            环境矩阵值（分段定位输入）
        em: np.ndarray, shape=(nloc, nnei, 4), dtype=float32
            环境矩阵（4 分量）
        dy: np.ndarray, shape=(nloc, 4, L), dtype=float32
            上游梯度
        descriptor: np.ndarray, shape=(nloc, 4, L), dtype=float32
            前向输出（仅用 shape[2] 提取 L，值不参与计算）
        split_count: int, 1 或 2，1=单核，2=双核并行（TBE 框架注入属性，非 REG_OP ATTR）
        split_index: int, 0 或 1，0=AI Core，1=Vector Core（需 split_count=2）

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        tuple (dy_dem_x, dy_dem):
            dy_dem_x: np.ndarray, shape=(nloc*nnei, 1), dtype=float32 - 对 em_x 的梯度
            dy_dem:   np.ndarray, shape=(nloc, nnei, 4), dtype=float32 - 对 em 的梯度
    """
    # 强制 float32
    table = np.asarray(table, dtype=np.float32)
    table_info = np.asarray(table_info, dtype=np.float32)
    em_x = np.asarray(em_x, dtype=np.float32)
    em = np.asarray(em, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)
    descriptor = np.asarray(descriptor, dtype=np.float32)

    nloc = em.shape[0]
    nnei = em.shape[1]
    last_layer_size = descriptor.shape[2]
    size_align64 = (last_layer_size + 63) // 64 * 64

    # table_info 分段参数
    lower = float(table_info[0])
    upper = float(table_info[1])
    vmax = float(table_info[2])
    stride0 = float(table_info[3])
    stride1 = float(table_info[4])

    first_stride = int(np.floor((upper - lower) / stride0))
    max_tbl_idx = first_stride + int(np.floor((vmax - upper) / stride1)) - 1

    # split 切分
    if split_count == 2:
        nloc_split = (nloc + 1) // 2
        if split_index == 0:
            loc_start, loc_end = 0, nloc_split
        else:
            loc_start, loc_end = nloc_split, nloc
    else:
        loc_start, loc_end = 0, nloc

    # 输出初始化
    dy_dem_x = np.zeros((nloc * nnei, 1), dtype=np.float32)
    dy_dem = np.zeros((nloc, nnei, 4), dtype=np.float32)

    # 将 em_x 展平为一维（nloc*nnei,）
    em_x_flat = em_x.reshape(nloc * nnei)

    for loc in range(loc_start, loc_end):
        # 末邻居值（用于识别重复填充）
        last_nei_val = (
            em_x_flat[(loc + 1) * nnei - 1]
            if loc < nloc - 1 or nloc > 0
            else np.float32(0.0)
        )

        for nei in range(nnei):
            x = float(em_x_flat[loc * nnei + nei])

            # 分段定位（三段式）
            if x >= vmax:
                table_idx = max_tbl_idx
                x_local = 0.0
            elif x >= upper:
                table_idx = first_stride + int(np.floor((x - upper) / stride1))
                x_local = (x - upper) - int(np.floor((x - upper) / stride1)) * stride1
            elif x >= lower:
                table_idx = int(np.floor((x - lower) / stride0))
                x_local = (x - lower) - int(np.floor((x - lower) / stride0)) * stride0
            else:
                # x < lower：table_idx=0，x_local 为负值（依赖系数外推）
                table_idx = 0
                x_local = x - lower

            table_idx = max(0, min(table_idx, table.shape[0] - 1))

            # 加载 6 段系数（每段 size_align64，取前 last_layer_size）
            row = table[table_idx]
            a0 = row[0 * size_align64 : 1 * size_align64][:last_layer_size]
            a1 = row[1 * size_align64 : 2 * size_align64][:last_layer_size]
            a2 = row[2 * size_align64 : 3 * size_align64][:last_layer_size]
            a3 = row[3 * size_align64 : 4 * size_align64][:last_layer_size]
            a4 = row[4 * size_align64 : 5 * size_align64][:last_layer_size]
            a5 = row[5 * size_align64 : 6 * size_align64][:last_layer_size]

            # 前向值 res = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
            res = (
                a0
                + a1 * x_local
                + a2 * x_local**2
                + a3 * x_local**3
                + a4 * x_local**4
                + a5 * x_local**5
            )

            # 梯度 grad = a1 + 2*a2*x + 3*a3*x^2 + 4*a4*x^3 + 5*a5*x^4
            grad = (
                a1
                + 2 * a2 * x_local
                + 3 * a3 * x_local**2
                + 4 * a4 * x_local**3
                + 5 * a5 * x_local**4
            )

            # dy_dem: 对 em 的梯度（4 分量），沿 size 累加
            # dy_dem[loc, nei, c] = sum_size res * dy[loc, c, size]
            for c in range(4):
                dy_dem[loc, nei, c] = np.sum(res * dy[loc, c, :])

            # dy_dem_x: 对 em_x 的梯度（标量），沿 size 累加
            # em_dy_dot[size] = sum_c em[loc,nei,c] * dy[loc,c,size]
            em_vec = em[loc, nei, :]  # (4,)
            em_dy_dot = np.zeros(last_layer_size, dtype=np.float32)
            for c in range(4):
                em_dy_dot += em_vec[c] * dy[loc, c, :]

            dy_dem_x_val = np.sum(grad * em_dy_dot)

            # 末邻居特殊处理：如果当前邻居值等于最后一个邻居值，且是重复填充
            if nei == nnei - 1 and x == last_nei_val:
                # 统计等于 last_nei_val 的连续邻居数（从末尾向前）
                count_last = 0
                for k in range(nnei - 1, -1, -1):
                    if em_x_flat[loc * nnei + k] == last_nei_val:
                        count_last += 1
                    else:
                        break
                if count_last > 1:
                    # 将之前的重复邻居梯度清零，集中在末邻居处累加
                    for k in range(nei - count_last + 1, nei):
                        dy_dem_x[loc * nnei + k] = 0.0
                        dy_dem[loc, k, :] = 0.0
                    dy_dem_x_val *= count_last
                    dy_dem[loc, nei, :] *= count_last

            dy_dem_x[loc * nnei + nei] = dy_dem_x_val

    # split_count=2 时只返回当前 split 的部分
    if split_count == 2:
        nloc_split = (nloc + 1) // 2
        if split_index == 0:
            out_dy_dem_x = dy_dem_x[: nloc_split * nnei]
            out_dy_dem = dy_dem[:nloc_split]
        else:
            out_dy_dem_x = dy_dem_x[nloc_split * nnei :]
            out_dy_dem = dy_dem[nloc_split:]
        return out_dy_dem_x.reshape(-1, 1), out_dy_dem

    return dy_dem_x.reshape(-1, 1), dy_dem
