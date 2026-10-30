/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ADD_LORA_TILING_H_
#define _ADD_LORA_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddLoraTilingData {
  uint32_t usedCoreNum = 0;
  uint32_t layer = 0;
  uint32_t batch = 0;
  uint32_t H1 = 0;
  uint32_t H2 = 0;
  uint32_t R = 0;
  uint32_t wBatch = 0;
  uint32_t layer_idx = 0;
  uint32_t y_offset = 0;
  float scale = 0;
  uint32_t y_slice_size = 0;
  uint32_t taskNumPerCore = 0;
  uint32_t addLoraFlag = 0;
  uint32_t y_column = 0;
  uint32_t H2PerCore = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)  \
  __ubuf__ tilingStruct* tilingDataPointer =                                 \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)     \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  AddLoraTilingData tilingData;                                               \
  INIT_TILING_DATA(AddLoraTilingData, tilingDataPointer, tilingPointer);  \
  (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                              \
  (tilingData).layer = tilingDataPointer->layer;                          \
  (tilingData).batch = tilingDataPointer->batch;                  \
  (tilingData).H1 = tilingDataPointer->H1;                  \
  (tilingData).H2 = tilingDataPointer->H2;                  \
  (tilingData).R = tilingDataPointer->R;                  \
  (tilingData).wBatch = tilingDataPointer->wBatch;                  \
  (tilingData).layer_idx = tilingDataPointer->layer_idx;                  \
  (tilingData).y_offset = tilingDataPointer->y_offset;                  \
  (tilingData).y_slice_size = tilingDataPointer->y_slice_size;                  \
  (tilingData).taskNumPerCore= tilingDataPointer->taskNumPerCore;                  \
   (tilingData).addLoraFlag= tilingDataPointer->addLoraFlag;                  \
  (tilingData).scale= tilingDataPointer->scale;                  \
  (tilingData).y_column = tilingDataPointer->y_column;          \
  (tilingData).H2PerCore = tilingDataPointer->H2PerCore;

#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_GAMMA half
#define DTYPE_Y half
#define DTYPE_X half

#endif