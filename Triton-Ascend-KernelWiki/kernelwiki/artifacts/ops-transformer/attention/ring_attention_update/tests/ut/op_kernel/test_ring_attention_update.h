/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RING_ATTENTION_UPDATE_TILING_H_
#define _RING_ATTENTION_UPDATE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct RingAttentionUpdateTilingData {
  int64_t batchSize;
  int64_t headNum;
  int64_t seqNum;
  int64_t headDim;
  int64_t softmaxTailSize;

  int64_t coreNum;
  int64_t coreNumGroup;
  int64_t bnNumGroup;
  int64_t seqNumCoreEach;
  int64_t seqNumCoreTail;
  int64_t seqNumLoopEach;
  int64_t headNumLoopEach;
  int64_t headDimLoopEach;

  int64_t batchSizeCoreEach;
  int64_t batchSizeCoreTail;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)  \
  __ubuf__ tilingStruct* tilingDataPointer =                                 \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)     \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  RingAttentionUpdateTilingData tilingData;                                               \
  INIT_TILING_DATA(RingAttentionUpdateTilingData, tilingDataPointer, tilingPointer);  \
  (tilingData).batchSize = tilingDataPointer->batchSize;                              \
  (tilingData).headNum = tilingDataPointer->headNum;                          \
  (tilingData).seqNum = tilingDataPointer->seqNum;                  \
  (tilingData).headDim = tilingDataPointer->headDim;            \
  (tilingData).softmaxTailSize = tilingDataPointer->softmaxTailSize;            \
  (tilingData).coreNum = tilingDataPointer->coreNum;              \
  (tilingData).coreNumGroup = tilingDataPointer->coreNumGroup;            \
  (tilingData).bnNumGroup = tilingDataPointer->bnNumGroup;            \
  (tilingData).seqNumCoreEach = tilingDataPointer->seqNumCoreEach;            \
  (tilingData).seqNumCoreTail = tilingDataPointer->seqNumCoreTail;            \
  (tilingData).seqNumLoopEach = tilingDataPointer->seqNumLoopEach;            \
  (tilingData).headNumLoopEach = tilingDataPointer->headNumLoopEach;            \
  (tilingData).headDimLoopEach = tilingDataPointer->headDimLoopEach;           \
  (tilingData).batchSizeCoreEach = tilingDataPointer->batchSizeCoreEach;           \
  (tilingData).batchSizeCoreTail = tilingDataPointer->batchSizeCoreTail;
#endif