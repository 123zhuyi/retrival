/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
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
 * \file minimum.h
 * \brief
 */
#ifndef MINIMUM_H
#define MINIMUM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "minimum_tiling_data.h"
#include "minimum_tiling_key.h"

namespace NsMinimum {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Minimum {
public:
    __aicore__ inline Minimum(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MinimumTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T>
__aicore__ inline void Minimum<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MinimumTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    inputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline void Minimum<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Minimum<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void Minimum<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        auto tmpX = tmpBuf0.Get<float>();
        auto tmpY = tmpBuf1.Get<float>();
        AscendC::Cast(tmpX, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmpY, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Min(tmpX, tmpX, tmpY, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Cast(zLocal, tmpX, AscendC::RoundMode::CAST_ROUND, this->processDataNum);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        auto tmpX = tmpBuf0.Get<half>();
        auto tmpY = tmpBuf1.Get<half>();
        AscendC::Cast(tmpX, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmpY, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Min(tmpX, tmpX, tmpY, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Cast(zLocal, tmpX, AscendC::RoundMode::CAST_RINT, this->processDataNum);
    } else {
        /* float/half/int16/int32 等直接 Min */
        AscendC::Min(zLocal, xLocal, yLocal, this->processDataNum);
    }

    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Minimum<T>::Process()
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

} // namespace NsMinimum
#endif // Minimum_H