/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_GROUPED_BIAS_ADD_GRAD_H_
#define _TEST_GROUPED_BIAS_ADD_GRAD_H_

#include "kernel_tiling/kernel_tiling.h"

struct GroupedBiasAddGradTilingData {
  uint32_t usedCoreNum;
  uint32_t normalCoreNum;
  uint32_t normalCoreProcessNum;
  uint32_t tailCoreProcessNum;
  int64_t wsUnitNum;
  int64_t dimG;
  int64_t dimC;
  int64_t dimH;
  int64_t dimGB;
  uint32_t baseH;
  uint32_t baseC;
  uint32_t loopCNum;
  int32_t groupIdxType;
};

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                  \
  GroupedBiasAddGradTilingData tilingData;                                          \
  INIT_TILING_DATA(GroupedBiasAddGradTilingData, tilingDataPointer, tilingPointer); \
  (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                        \
  (tilingData).normalCoreNum = tilingDataPointer->normalCoreNum;                    \
  (tilingData).normalCoreProcessNum = tilingDataPointer->normalCoreProcessNum;      \
  (tilingData).tailCoreProcessNum = tilingDataPointer->tailCoreProcessNum;          \
  (tilingData).wsUnitNum = tilingDataPointer->wsUnitNum;                            \
  (tilingData).dimG = tilingDataPointer->dimG;                                      \
  (tilingData).dimC = tilingDataPointer->dimC;                                      \
  (tilingData).dimH = tilingDataPointer->dimH;                                      \
  (tilingData).dimGB = tilingDataPointer->dimGB;                                    \
  (tilingData).baseH = tilingDataPointer->baseH;                                    \
  (tilingData).baseC = tilingDataPointer->baseC;                                    \
  (tilingData).loopCNum = tilingDataPointer->loopCNum;                              \
  (tilingData).groupIdxType = tilingDataPointer->groupIdxType;

#endif  // _TEST_GROUPED_BIAS_ADD_GRAD_H_
