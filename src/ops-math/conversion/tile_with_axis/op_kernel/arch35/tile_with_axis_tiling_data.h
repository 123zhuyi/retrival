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
 * \file tile_with_axis_tiling_data.h
 * \brief TileWithAxis TilingData 结构体定义
 *
 * 设计依据: DESIGN.md v1.6 3.3 节
 * 3D 模型: inShape=[outerDim,1,rowLength], outShape=[outerDim,tiles,rowLength]
 * rowLength = axisDim * innerDim
 *
 * 字段分为两个层次:
 *   A. LayoutTransform 单切分公共字段
 *   B. TileWithAxis 特有字段
 */

#ifndef _TILE_WITH_AXIS_TILING_DATA_H_
#define _TILE_WITH_AXIS_TILING_DATA_H_

#include <cstdint>

struct TileWithAxisTilingData {
    // ==== A. LayoutTransform 单切分公共字段 ====
    int64_t  inShape[3];        // 输入展平 shape: [outerDim, 1, rowLength]
    int64_t  outShape[3];       // 输出展平 shape: [outerDim, tiles, rowLength]
    uint64_t totalCount;        // UB 切分块总数（基于 outShape 计算）
    uint64_t perCoreCount;      // 每核处理的块数（totalCount 均分给 coreNum 个核）
    uint8_t  ubAxis;            // UB 切分轴（0=outerDim, 1=tiles, 2=rowLength）
    uint32_t ubFactor;          // 切分因子（沿 ubAxis 每次搬运该轴多少个元素）
    uint32_t bufferSize;        // UB buffer 字节大小（ping-pong 单个 buffer 的字节数）

    // ==== B. TileWithAxis 特有字段 ====
    int64_t  tiles;             // 复制次数
    int64_t  rowLength;         // 折叠后每行长度 = axisDim * innerDim
};

#endif // _TILE_WITH_AXIS_TILING_DATA_H_
