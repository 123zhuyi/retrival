/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _FAST_OP_TEST_CIRCULAR_PAD_TILING_H_
#define _FAST_OP_TEST_CIRCULAR_PAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
#define __CCE_KT_TEST__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                   \
    CircularPadCommonTilingData tilingData;                                          \
    INIT_TILING_DATA(CircularPadCommonTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).inputH = tilingDataPointer->inputH;                                 \
    (tilingData).inputW = tilingDataPointer->inputW;                                 \
    (tilingData).outputH = tilingDataPointer->outputH;                               \
    (tilingData).outputW = tilingDataPointer->outputW;                               \
    (tilingData).left = tilingDataPointer->left;                                     \
    (tilingData).right = tilingDataPointer->right;                                   \
    (tilingData).top = tilingDataPointer->top;                                       \
    (tilingData).bottom = tilingDataPointer->bottom;                                 \
    (tilingData).front = tilingDataPointer->front;                                   \
    (tilingData).back = tilingDataPointer->back;                                     \
    (tilingData).inputL = tilingDataPointer->inputL;                                 \
    (tilingData).outputL = tilingDataPointer->outputL;                               \
    (tilingData).perCoreTaskNum = tilingDataPointer->perCoreTaskNum;                 \
    (tilingData).tailTaskNum = tilingDataPointer->tailTaskNum;                       \
    (tilingData).workspaceLen = tilingDataPointer->workspaceLen;

#endif // _FAST_OP_TEST_CIRCULAR_PAD_TILING_H_
