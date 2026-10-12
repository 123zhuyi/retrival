/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _IS_INF_TILING_H_
#define _IS_INF_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define __CCE_UT_TEST__

#pragma pack(1)

struct IsInfTilingData {
  uint32_t totalDataCount;
  uint32_t usableUbSize;
  uint32_t needCoreNum;
  uint32_t perCoreDataCount;
  uint32_t tailDataCoreNum;
  uint32_t lastCoreDataCount;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)  \
  __ubuf__ tilingStruct* tilingDataPointer =                                 \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)     \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  IsInfTilingData tilingData;                                               \
  INIT_TILING_DATA(IsInfTilingData, tilingDataPointer, tilingPointer);  \
  (tilingData).totalDataCount = tilingDataPointer->totalDataCount;                              \
  (tilingData).usableUbSize = tilingDataPointer->usableUbSize;                          \
  (tilingData).needCoreNum = tilingDataPointer->needCoreNum;                  \
  (tilingData).perCoreDataCount = tilingDataPointer->perCoreDataCount;            \
  (tilingData).tailDataCoreNum = tilingDataPointer->tailDataCoreNum;              \
  (tilingData).lastCoreDataCount = tilingDataPointer->lastCoreDataCount;            
  
#endif