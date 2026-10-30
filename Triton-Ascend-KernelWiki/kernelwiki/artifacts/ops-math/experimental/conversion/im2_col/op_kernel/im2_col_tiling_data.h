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
 * \file im2_col_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_
#define _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_

struct Im2ColTilingData {
    // 输入输出维度
    uint32_t N;
    uint32_t C;
    uint32_t H;
    uint32_t W;

    // 卷积参数
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t dilation_h;
    int32_t dilation_w;

    // 输出维度
    uint32_t out_H;
    uint32_t out_W;
    uint32_t L;
    uint32_t output_channels;

    // 核心Tiling参数
    uint32_t input_elements;
    uint32_t output_elements;
    uint32_t total_output_elements;
    uint32_t base_elements_per_core;
    uint32_t big_core_num;
    uint32_t tile_element_num;
    uint32_t aligned_element_size;
};
#endif