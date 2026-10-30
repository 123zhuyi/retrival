/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file clip_by_norm_no_div_sum_tiling_data.h
 * \brief clip_by_norm_no_div_sum TilingData
 */
#pragma once
#include <cstdint>

constexpr int64_t kMaxInputSlots = 4;  // 4 输入: x, greater_zeros, select_ones, maximum_ones
constexpr int64_t kMaxOutputSlots = 1; // 1 输出: y
constexpr int64_t kPhysNodes = 5;      // 物理存活节点 P = 5（4 输入 + 1 输出 UB buffer）

struct SplitResult {
    int64_t axis;     // UB 切分轴
    int64_t a_i;      // 内轴 tile 大小（元素数）
    int64_t a_o;      // 外轴 tile 数
    int64_t a_i_tail; // 末块大小（元素数）
};

struct MultiCoreResult {
    int64_t num_cores;   // 参与计算的核数
    int64_t total_tiles; // tile 总数
    int64_t tiles_main;  // 每核主 tile 数
    int64_t cores_tail;  // 多处理一个 tile 的核数
};

template <int64_t kRank>
struct ClipByNormNoDivSumTilingData {
    SplitResult split;                              // UB 切分结果（来源：FindSplitAxis）
    MultiCoreResult multicore;                      // 多核切分结果（来源：MultiCoreSplit）
    int64_t rank;                                   // 实际有效 rank（去 1 补 1 后）
    int64_t per_buf_bytes;                          // 单 buffer 字节数 = (UB_AVAIL / P) & ~31
    int64_t max_bro_shape[kRank];                   // 广播后各维大小（坐标系）
    int64_t num_inputs;                             // 输入张量数
    int64_t num_outputs;                            // 输出张量数
    int64_t input_shapes[kMaxInputSlots][kRank];    // 各输入补 1 后的 shape
    int64_t input_strides[kMaxInputSlots][kRank];   // 各输入 GM stride（broadcast 轴 = 0）
    int64_t output_shapes[kMaxOutputSlots][kRank];  // 各输出补 1 后的 shape
    int64_t output_strides[kMaxOutputSlots][kRank]; // 各输出的 GM stride
};
