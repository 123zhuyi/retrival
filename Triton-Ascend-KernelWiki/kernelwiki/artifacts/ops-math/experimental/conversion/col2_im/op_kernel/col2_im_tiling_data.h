/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file col2_im_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __COL_2_IM_TILLING_DATA_H__
#define __COL_2_IM_TILLING_DATA_H__

constexpr int32_t MAX_USE_CORE_NUM = 32; // 设置合理的最大核数

struct Col2ImTilingData {
    // ========== 基础维度参数 ==========
    int32_t N;
    int32_t C;
    int32_t H;
    int32_t W;

    // ========== 卷积参数 ==========
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_val;
    int32_t padding_val;
    int32_t dilation_val;

    // ========== 输入列格式维度 ==========
    int32_t input_channels; // C * kH * kW
    int32_t L;              // out_H * out_W
    int32_t out_H;
    int32_t out_W;

    // ========== 元素数量 ==========
    uint64_t input_elements;  // 输入col元素总数
    uint64_t output_elements; // 输出图像元素总数

    // ========== 核间划分信息（移除数组，改为计算参数） ==========
    uint32_t total_output_pixels; // N*C*H*W
    int32_t base_pixels_per_core; // 每个core的基础像素数
    int32_t big_core_num;         // 需要+1个像素的core数量

    // ========== 核内划分信息（移除数组） ==========
    int32_t tile_pixel_num;    // 每个tile处理的像素数
    int32_t aligned_tile_size; // 对齐后的tile大小

    // ========== 向量化参数 ==========
    uint32_t vec_elements; // 向量长度（如8/16/32）
};
#endif
