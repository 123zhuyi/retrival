/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _FAST_OP_TEST_FILL_DIAGONAL_V2_TILING_H_
#define _FAST_OP_TEST_FILL_DIAGONAL_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
struct FillDiagonalV2TilingData {
    uint64_t totalLength;
    uint64_t step;
    uint64_t end;
    uint64_t ubSize;
    uint64_t blockLength;
    uint64_t lastBlockLength;
};

#define DTYPE_X float

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                                          \
  FillDiagonalV2TilingData tilingData;                                                                      \
  INIT_TILING_DATA(FillDiagonalV2TilingData, tilingDataPointer, tilingPointer);                             \
  (tilingData).totalLength = tilingDataPointer->totalLength;                                                \
  (tilingData).step = tilingDataPointer->step;                                                              \
  (tilingData).end = tilingDataPointer->end;                                                                \
  (tilingData).ubSize = tilingDataPointer->ubSize;                                                          \
  (tilingData).blockLength = tilingDataPointer->blockLength;                                                \
  (tilingData).lastBlockLength = tilingDataPointer->lastBlockLength;
#endif // _FAST_OP_TEST_FILL_DIAGONAL_TILING_H_