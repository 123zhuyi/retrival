/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liang Yanglin <@liang-yanglin>
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
 * \file cast_v2.h
 * \brief
 */
#ifndef __CAST_V2_H__
#define __CAST_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cast_v2_tiling_data.h"
#include "cast_v2_tiling_key.h"

namespace NsCastV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Z>
class CastV2 {
public:
    __aicore__ inline CastV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const CastV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Z> zGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t mainCoreNum;
    int64_t globalBufferIndex;
};

template <typename TYPE_X, typename TYPE_Z>
__aicore__ inline void CastV2<TYPE_X, TYPE_Z>::Init(GM_ADDR x, GM_ADDR z, const CastV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();

    this->globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->mainCoreNum = tilingData->tailBlockNum;
    if (coreNum < this->mainCoreNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        this->globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (AscendC::GetBlockIdx() - this->mainCoreNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Z));
}

template <typename TYPE_X, typename TYPE_Z>
__aicore__ inline void CastV2<TYPE_X, TYPE_Z>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Z>
__aicore__ inline void CastV2<TYPE_X, TYPE_Z>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();
    AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
    outQueueZ.FreeTensor(zLocal);
}

template <typename TYPE_X, typename TYPE_Z>
__aicore__ inline void CastV2<TYPE_X, TYPE_Z>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();
    if constexpr (std::is_same_v<TYPE_X, int32_t> && std::is_same_v<TYPE_Z, half>) {
        half scale = 1.0f;
        AscendC::SetDeqScale(scale);
        AscendC::Cast(zLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
    } else {
        AscendC::Cast(zLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
    }
    outQueueZ.EnQue<TYPE_Z>(zLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Z>
__aicore__ inline void CastV2<TYPE_X, TYPE_Z>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsCastV2
#endif // CAST_V2_H