/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file test_pows.h
 * \brief
 */

#ifndef _POWS_TILING_H_
#define _POWS_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct PowsTilingData {
    int64_t mainCoreLoopNum;
    int64_t mainCoreTailLength;
    int64_t tailCoreLoopNum;
    int64_t tailCoreTailLength;
    int64_t realCoreNum;
    int64_t numPerCore;
    int64_t tilingKey;
    int64_t bufSize;
    int64_t dataLength;
    int64_t blockSize;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
    PowsTilingData tilingData;                                               \
    INIT_TILING_DATA(PowsTilingData, tilingDataPointer, tilingPointer);      \
    (tilingData).mainCoreLoopNum = tilingDataPointer->mainCoreLoopNum;       \
    (tilingData).mainCoreTailLength = tilingDataPointer->mainCoreTailLength; \
    (tilingData).tailCoreLoopNum = tilingDataPointer->tailCoreLoopNum;       \
    (tilingData).tailCoreTailLength = tilingDataPointer->tailCoreTailLength; \
    (tilingData).realCoreNum = tilingDataPointer->realCoreNum;               \
    (tilingData).numPerCore = tilingDataPointer->numPerCore;                 \
    (tilingData).tilingKey = tilingDataPointer->tilingKey;                   \
    (tilingData).blockSize = tilingDataPointer->blockSize;                   \
    (tilingData).bufSize = tilingDataPointer->bufSize;                       \
    (tilingData).dataLength = tilingDataPointer->dataLength;
#endif