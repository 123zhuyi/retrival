/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_SINKHORN_TILING_H_
#define UT_SINKHORN_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct SinkhornTilingDataUT {
  uint64_t formerNum;            // former 数量
  uint64_t formerRow;            // former cost行数
  uint64_t formerLength;         // former cost总长

  uint64_t formerTileNum;        // former Tile数量
  uint64_t formerLastTileRow;    // fomer last Tile行数
  uint64_t formerLastTileLength; // fomer last Tile长度

  uint64_t tailNum;              // tail 数量
  uint64_t tailRow;              // tail cost行数
  uint64_t tailLength;           // tail cost总长

  uint64_t tailTileNum;          // tail Tile数量
  uint64_t tailLastTileRow;      // tail last Tile行数
  uint64_t tailLastTileLength;   // tail last Tile长度

  uint64_t tileRow;              // Tile行数(非Last)
  uint64_t tileLength;           // Tile长度(非Last)

  uint64_t totalRow;             // 总行数
  uint64_t totalCol;             // 总列数
  uint64_t totalColAligned;      // 对齐后的总列数

  float tol;                     // 误差
};

typedef SinkhornTilingDataUT SinkhornTilingData;
#define TILING_DATA_STRUCT_NAME SinkhornTilingDataUT
#define GET_TILING_DATA(tiling_data, tiling)                     \
  TILING_DATA_STRUCT_NAME tiling_data;                           \
  {                                                              \
    auto tempTilingGM = (__gm__ uint32_t*)tiling;                \
    auto tempTiling = (uint32_t*)&tiling_data;                   \
    for (int i = 0;                                              \
      i < sizeof(TILING_DATA_STRUCT_NAME) / sizeof(uint32_t);    \
      ++i, ++tempTilingGM, ++tempTiling)                         \
    {                                                            \
      *tempTiling = *tempTilingGM;                               \
    }                                                            \
  }

#endif // UT_SINKHORN_TILING_H_
