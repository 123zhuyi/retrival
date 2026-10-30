/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILING_DATA_DEF_H
#define TILING_DATA_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__


struct ConcatDV2TilingDataUT {
    uint32_t elePerLoop = 0;
    int64_t elePercore = 0;
    int64_t ubLoop = 0;
    int64_t eleTailCore = 0;
    int64_t ubLoopTail = 0;
    int64_t sameDimSize = 0;
    int64_t endTensorIdx[48] = {0};
    int64_t endTensorOffset[48] = {0};

};

inline void InitConcatDV2TilingData(uint8_t* tiling, ConcatDV2TilingDataUT* const_data)
{
    memcpy(const_data, tiling, sizeof(ConcatDV2TilingDataUT));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    ConcatDV2TilingDataUT tilingData;              \
    InitConcatDV2TilingData(tilingPointer, &tilingData)
#endif