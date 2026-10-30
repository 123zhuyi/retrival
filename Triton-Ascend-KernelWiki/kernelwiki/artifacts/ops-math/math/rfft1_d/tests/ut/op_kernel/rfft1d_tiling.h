// /**
//  * This program is free software, you can redistribute it and/or modify it.
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This file is a part of the CANN Open Software.
//  * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
//  * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
//  * the software repository for the full text of the License.
//  */
// /*!
//  * \file rfft1d_tiling.h
//  * \brief
//  */

// #ifndef _GE_RFFT1D_TILING_H_
// #define _GE_RFFT1D_TILING_H_

// #include "kernel_tiling/kernel_tiling.h"

// #define DT_BF16 bfloat16_t
// #define ORIG_DTYPE_START DT_BF16
// #define __CCE_UT_TEST__

// #pragma pack(1)

// struct Rfft1DTilingData {
//     int32_t length;
//     uint8_t isBluestein;
//     int32_t lengthPad;
//     uint32_t factors[3] = {0, 0, 0};
//     uint32_t prevRadices[3] = {0, 0, 0};
//     uint32_t nextRadices[3] = {0, 0, 0};
//     uint32_t prevRadicesAlign[3] = {0, 0, 0};
//     int32_t outLength;
//     uint32_t tailSize;
//     uint32_t batchesPerCore;
//     uint32_t leftOverBatches;
//     uint32_t tmpLenPerBatch;
//     uint32_t tmpSizePerBatch;
//     uint32_t matmulTmpsLen;
//     uint32_t matmulTmpsSize;
//     int32_t normal;
//     uint32_t dftRealOverallSize;
//     uint32_t dftImagOverallSize;
//     uint32_t twiddleOverallSize;
//     uint32_t fftMatrOverallSize;
//     uint32_t dftRealOffsets[3] = {0, 0, 0};
//     uint32_t dftImagOffsets[3] = {0, 0, 0};
//     uint32_t twiddleOffsets[3] = {0, 0, 0};
// };

// #pragma pack()

// #ifdef __NPU_TILING__
// inline [aicore] void InitTilingData(const __gm__ uint8_t* tiling, Rfft1DTilingData* constData) {
//     const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
//     uint32_t* dst = (uint32_t*)constData;
//     for (auto i = 0; i < sizeof(Rfft1DTilingData) / 4; i++)
//         *(dst + i) = *(src + i);
// }
// #else
// inline void InitTilingData(uint8_t* tiling, Rfft1DTilingData* constData)
// {
//     memcpy(constData, tiling, sizeof(Rfft1DTilingData));
// }
// #endif

// #define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
//     tilingStruct tilingData;                                             \
//     InitTilingData(tilingArg, &tilingData)

// #define GET_TILING_DATA(tilingData, tilingArg) \
//     Rfft1DTilingData tilingData;               \
//     InitTilingData(tilingArg, &tilingData)

// #define DTYPE_X float
// #define DTYPE_Y float

// #endif