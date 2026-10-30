/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILING_DATA_DEF_H
#define TILING_DATA_DEF_H

#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

inline __aicore__ int32_t AlignDiv32(int32_t n)
{
    return ((n + 31) & ~31) / 32;
}

#define COPY_ARR(arrA, arrB, count)        \
    for (uint16_t i = 0; i < count; i++) { \
        arrA[i] = arrB[i];                 \
    }

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#ifdef __CCE_KT_TEST__
#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);
#else
#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)                  \
    __ubuf__ uint8_t* tilingUbPointer = (__ubuf__ uint8_t*)get_imm(0);                    \
    copy_gm_to_ubuf(                                                                      \
        ((__ubuf__ uint8_t*)(tilingUbPointer)), ((__gm__ uint8_t*)(tilingPointer)), 0, 1, \
        AlignDiv32(sizeof(tilingStruct)), 0, 0);                                          \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingUbPointer);                \
    pipe_barrier(PIPE_ALL);
#endif

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
    PadV4GradTilingData tilingData;                                          \
    INIT_TILING_DATA(PadV4GradTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).batch = tilingDataPointer->batch;                           \
    (tilingData).channel = tilingDataPointer->channel;                       \
    (tilingData).height = tilingDataPointer->height;                         \
    (tilingData).width = tilingDataPointer->width;                           \
    (tilingData).alignHeight = tilingDataPointer->alignHeight;               \
    (tilingData).alignWidth = tilingDataPointer->alignWidth;                 \
    (tilingData).outHeight = tilingDataPointer->outHeight;                   \
    (tilingData).outWidth = tilingDataPointer->outWidth;                     \
    (tilingData).alignOutHeight = tilingDataPointer->alignOutHeight;         \
    (tilingData).alignOutWidth = tilingDataPointer->alignOutWidth;           \
    (tilingData).hPad1 = tilingDataPointer->hPad1;                           \
    (tilingData).hPad2 = tilingDataPointer->hPad2;                           \
    \  
    (tilingData)                                                             \
        .wPad1 = tilingDataPointer->wPad1;                                   \
    (tilingData).wPad2 = tilingDataPointer->wPad2;                           \
    (tilingData).blockNum = tilingDataPointer->blockNum;                     \
    \        
    (tilingData)                                                             \
        .ubFactorElement = tilingDataPointer->ubFactorElement;               \
    \                     
    (tilingData)                                                             \
        .ncPerCore = tilingDataPointer->ncPerCore;                           \
    \         
    (tilingData)                                                             \
        .tailNC = tilingDataPointer->tailNC;                                 \
    \   
    (tilingData)                                                             \
        .tilingKey = tilingDataPointer->tilingKey;                           \
    \         
    (tilingData)                                                             \
        .workspacePerCore = tilingDataPointer->workspacePerCore;
#endif // TILING_DATA_DEF_H
