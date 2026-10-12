/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _FAST_OP_TEST_SIGNBIT_H_
#define _FAST_OP_TEST_SIGNBIT_H_

#include "kernel_tiling/kernel_tiling.h"

struct SignbitTilingData {
    int64_t dim0;
    int32_t coreNum;
    int32_t ubFormer;
    int64_t blockFormer;
    int64_t blockNum;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
    int64_t elemNum;
    uint64_t scheMode;
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
  SignbitTilingData tilingData;                                                     \
  INIT_TILING_DATA(SignbitTilingData, tilingDataPointer, tilingPointer);            \
  (tilingData).dim0 = tilingDataPointer->dim0;                      \
  (tilingData).coreNum = tilingDataPointer->coreNum;                          \
  (tilingData).ubFormer = tilingDataPointer->ubFormer;                              \
  (tilingData).blockFormer = tilingDataPointer->blockFormer;                    \
  (tilingData).blockNum = tilingDataPointer->blockNum;                        \
  (tilingData).ubLoopOfFormerBlock = tilingDataPointer->ubLoopOfFormerBlock;         \
  (tilingData).ubLoopOfTailBlock = tilingDataPointer->ubLoopOfTailBlock;        \
  (tilingData).ubTailOfFormerBlock = tilingDataPointer->ubTailOfFormerBlock;                            \
  (tilingData).ubTailOfTailBlock = tilingDataPointer->ubTailOfTailBlock;                            \
  (tilingData).elemNum = tilingDataPointer->elemNum;                            \
  (tilingData).scheMode = tilingDataPointer->scheMode;
#endif // _FAST_OP_TEST_SIGNBIT_H_
