/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _FAST_OP_TEST_ANGLE_V2_H_
#define _FAST_OP_TEST_ANGLE_V2_H_

#include "kernel_tiling/kernel_tiling.h"

struct AngleV2TilingData {
  uint32_t totalLength = 0;
  uint32_t formerNum = 0;
  uint64_t tailNum = 0;
  uint32_t formerLength = 0;
  uint32_t tailLength = 0;
  uint32_t alignNum = 0;
  uint32_t totalLengthAligned = 0;
  uint32_t tileLength = 0;
  uint32_t dataPerRepeat = 0;
};

#define DTYPE_X int64_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                \
  AngleV2TilingData tilingData;                                                     \
  INIT_TILING_DATA(AngleV2TilingData, tilingDataPointer, tilingPointer);            \
  (tilingData).totalLength = tilingDataPointer->totalLength;                      \
  (tilingData).formerNum = tilingDataPointer->formerNum;                          \
  (tilingData).tailNum = tilingDataPointer->tailNum;                              \
  (tilingData).formerLength = tilingDataPointer->formerLength;                    \
  (tilingData).tailLength = tilingDataPointer->tailLength;                        \
  (tilingData).alignNum = tilingDataPointer->alignNum;                            \     
  (tilingData).totalLengthAligned = tilingDataPointer->totalLengthAligned;        \
  (tilingData).tileLength = tilingDataPointer->tileLength;                            \     
  (tilingData).dataPerRepeat = tilingDataPointer->dataPerRepeat;
#endif // _FAST_OP_TEST_ANGLE_V2_H_
