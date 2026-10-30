/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GE_STFT_TILING_H_
#define _GE_STFT_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct STFTTilingData {
    uint32_t tilingKey;
    uint32_t batch;
    uint32_t inputSize;
    uint32_t nfft;
    uint32_t hop;
    uint32_t frameCount;
    uint32_t frameCountAlign;
    uint32_t blkFrame;
    uint32_t matmulM;
    uint32_t sizePerRepeat;
    uint32_t blockNum;
    uint32_t aicBatchFactor;
    uint32_t aicBatchLoop;
    uint32_t aicTotalLen;
    uint32_t aicTailLen;
    uint32_t aicMatmulMCore;
    uint32_t aivBatchLoop;
    uint32_t aivTotalEvenRow;
    uint32_t aivTotalOddRow;
    uint32_t aivTailEvenRow;
    uint32_t aivTailOddRow;
    uint32_t aivWindowLoop;
    uint32_t aivGatherUbGap;
    uint32_t aicBatchTailIdx;
    uint32_t aivBatchTailIdx;
    uint32_t aicMTailIdx;
    uint32_t aivMTailIdx;
    uint32_t aicTailLoop;
    uint32_t aivTailLoop;
    uint32_t reservedData;
    TCubeTiling mmTilingData;
};

#pragma pack()

#pragma pack(1)

struct STFTPlanTilingData {
    uint32_t oneRowLen;
    uint32_t totalLine;
    uint32_t tailLine;
    uint32_t ubMaxLine;
    uint32_t totalInCol;
    uint32_t tailInCol;
    uint32_t tailBlockIdx;
    uint32_t batch;
    uint32_t matmulM;
    uint32_t matmulN;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, STFTTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(STFTTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, STFTTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(STFTTilingData));
}
#endif

#pragma pack(1)

struct STFTGeneralizedTilingData {
    uint32_t tilingKey;
    uint32_t batch;
    uint32_t inputSize;
    uint32_t nfft;
    uint32_t nfftAlign;
    uint32_t hopLength;
    uint32_t matmulM;
    uint32_t matmulN;
    uint32_t normalized;
    float rootNfft;
    uint32_t copyUBSize;
    uint32_t splitWindowSkipNum;
    uint32_t maskUBSize;
    uint32_t batchCoreNum;
    uint32_t batchCoreFactor;
    uint32_t batchTailCoreNum;
    uint32_t matmulMCoreNum;
    uint32_t matmulMCoreFactor;
    uint32_t matmulMTailCoreNum;
    uint32_t matmulNCoreNum;
    uint32_t matmulNCoreFactor;
    uint32_t matmulNTailCoreNum;
    uint32_t nFactorUbFormer;
    uint32_t nFactorUbTail;
    uint32_t nFactorUbLoop;
    uint32_t usedCoreNum;
    TCubeTiling mm0TilingData;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
    TCubeTiling mm3TilingData;
    STFTPlanTilingData planTilingData;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, STFTGeneralizedTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(STFTGeneralizedTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, STFTGeneralizedTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(STFTGeneralizedTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    STFTTilingData tiling_data;                  \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float

#endif
