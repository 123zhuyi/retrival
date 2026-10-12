/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_LINSPACE_H_
#define _TEST_LINSPACE_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct LinSpaceTilingData {
  float start = 0;
  float stop = 0;
  float scalar = 0;
  int64_t num = 0;
  int64_t matrixLen = 0;
  int64_t realCoreNum = 0;
  int64_t numPerCore = 0;
  int64_t tailNum = 0;
  int64_t tilingKey = 0;
  int64_t innerLoopNum = 0;
  int64_t innerLoopTail = 0;
  int64_t innerTailLoopNum = 0;
  int64_t innerTailLoopTail = 0;
  int64_t outerLoopNum = 0;
  int64_t outerLoopNumTail = 0;
  int64_t outerTailLoopNum = 0;
  int64_t outerTailLoopNumTail = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                        \
  LinSpaceTilingData tilingData;                                          \
  INIT_TILING_DATA(LinSpaceTilingData, tilingDataPointer, tilingPointer); \
  (tilingData).start = tilingDataPointer->start;                          \
  (tilingData).stop = tilingDataPointer->stop;                            \
  (tilingData).scalar = tilingDataPointer->scalar;                        \
  (tilingData).num = tilingDataPointer->num;                              \
  (tilingData).matrixLen = tilingDataPointer->matrixLen;                  \
  (tilingData).realCoreNum = tilingDataPointer->realCoreNum;              \
  (tilingData).numPerCore = tilingDataPointer->numPerCore;                \
  (tilingData).tailNum = tilingDataPointer->tailNum;                      \
  (tilingData).innerLoopNum = tilingDataPointer->innerLoopNum;            \
  (tilingData).innerLoopTail = tilingDataPointer->innerLoopTail;          \
  (tilingData).innerTailLoopNum = tilingDataPointer->innerTailLoopNum;    \
  (tilingData).innerTailLoopTail = tilingDataPointer->innerTailLoopTail;  \
  (tilingData).outerLoopNum = tilingDataPointer->outerLoopNum;            \
  (tilingData).outerLoopNumTail = tilingDataPointer->outerLoopNumTail;    \
  (tilingData).outerTailLoopNum = tilingDataPointer->outerTailLoopNum;    \
  (tilingData).outerTailLoopNumTail = tilingDataPointer->outerTailLoopNumTail;

#endif
