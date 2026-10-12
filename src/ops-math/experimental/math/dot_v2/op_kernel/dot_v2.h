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
 * \file dot_v2.h
 * \brief
 */
#ifndef DOT_V2_H
#define DOT_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dot_v2_tiling_data.h"
#include "dot_v2_tiling_key.h"

namespace NsDotV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class DotV2 {
public:
    __aicore__ inline DotV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, const DotV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(AscendC::LocalTensor<T> tmpTensor0);
    __aicore__ inline void Compute(int32_t progress, AscendC::LocalTensor<T> tmpTensor0);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQueue;

    TBuf<TPosition::VECCALC> tmpBuf0;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;
    GlobalTensor<T> workGM;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t workGmSize;
};

template <typename T>
__aicore__ inline void DotV2<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, const DotV2TilingData* tilingData)
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
    outputGMZ.SetGlobalBuffer((__gm__ T*)z, 1);
    this->workGmSize = tilingData->workGmSize; // dataCopy最少的元素数
    workGM.SetGlobalBuffer((__gm__ T*)workspace, this->workGmSize);
    if (AscendC::GetBlockIdx() == 0) {
        AscendC::InitGlobalMemory(workGM, workGmSize, (T)0.0);
    }
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmpQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline void DotV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void DotV2<T>::CopyOut(AscendC::LocalTensor<T> tmpTensor0)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    AscendC::SetAtomicAdd<T>();
    AscendC::DataCopy(workGM, tmpTensor0, this->workGmSize);
    AscendC::SetAtomicNone();
    AscendC::SyncAll();
}

template <typename T>
__aicore__ inline void DotV2<T>::Compute(int32_t progress, AscendC::LocalTensor<T> tmpTensor0)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    AscendC::LocalTensor<T> sharedTmpBuffer = tmpQueue.AllocTensor<T>();
    AscendC::Duplicate(zLocal, T(0.0f), this->processDataNum);
    AscendC::Mul(xLocal, xLocal, yLocal, this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(zLocal, xLocal, sharedTmpBuffer, this->processDataNum);
    PipeBarrier<PIPE_V>();
    T val = zLocal.GetValue(0);
    T accum = tmpTensor0.GetValue(0);
    tmpTensor0.SetValue(0, static_cast<T>(static_cast<float>(accum) + static_cast<float>(val)));
    outputQueueZ.FreeTensor(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
    tmpQueue.FreeTensor(sharedTmpBuffer);
}

template <typename T>
__aicore__ inline void DotV2<T>::Process()
{
    AscendC::SyncAll();
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    AscendC::LocalTensor<T> tmpTensor0 = tmpBuf0.Get<T>();
    AscendC::Duplicate(tmpTensor0, T(0.0f), this->processDataNum);
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i, tmpTensor0);
    }
    CopyOut(tmpTensor0);
    if (AscendC::GetBlockIdx() == 0) {
        outputGMZ.SetValue(0, workGM.GetValue(0));
    }
}
} // namespace NsDotV2
#endif // DotV2_H