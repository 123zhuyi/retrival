/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
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
 * \file pad_v2_tiling_data.h
 * \brief tiling data struct
 */
#ifndef _PADV2_TILING_DATA_H_
#define _PADV2_TILING_DATA_H_

struct PadV2TilingData {
    uint32_t totalLengthx;
    uint32_t small_tile_times;
    uint32_t big_tile_times;
    uint32_t small_tail_num;
    uint32_t big_tail_num;
    uint32_t big_core_num;
    uint32_t small_core_num;
    uint32_t small_tile_length;
    uint32_t big_tile_length;
    uint32_t core_tile_x1;
    int32_t dimNum;        // 真实维度数（<=4）
    int32_t dimarr[4];     // 原始 shape（不含 last dim）
    int32_t dimarrz[4];    // pad 后 shape
    int32_t pad[8];        // [l0,r0,l1,r1,...]
    int32_t bias[4];       // pad 后 bias
    int32_t orign_bias[4]; // 原始 bias
    int32_t idx[4];        // 临时 idx
    int64_t sumtimes;
    int64_t sumspace;
    int64_t rowz;
    int32_t value;    // pad 填充值
    int32_t mode;     // pad 填充值
    int32_t lpad;     // last dim 左 pad
    int32_t rpad;     // last dim 右 pad
    int32_t xlastdim; // 原始 last dim
};
#endif