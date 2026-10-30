/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
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
 * \file select_v3.h
 */
#ifndef SELECTV3_H
#define SELECTV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "select_v3_tiling_data.h"
#include "select_v3_tiling_key.h"

namespace NsSelectV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SelectV3 {
public:
    __aicore__ inline SelectV3(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR b, GM_ADDR z, const SelectV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<int8_t> inputGMB;
    GlobalTensor<T> outputGMZ;
    TBuf<TPosition::VECCALC> tmpBuf0;
    TBuf<TPosition::VECCALC> tmpBuf1;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T>
__aicore__ inline void SelectV3<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR b, GM_ADDR z, const SelectV3TilingData* tilingData)
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
    inputGMB.SetGlobalBuffer((__gm__ int8_t*)b + globalBufferIndex, this->coreDataNum);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueB, BUFFER_NUM, this->tileDataNum * sizeof(int8_t));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmpBuf0, (this->tileDataNum * sizeof(uint8_t) + 255) / 256 * 256 / 8);
    pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(half));
}

template <typename T>
__aicore__ inline void SelectV3<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::LocalTensor<int8_t> bLocal = inputQueueB.AllocTensor<int8_t>();
    AscendC::DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(sizeof(int8_t) * this->processDataNum), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int8_t> padParams{true, 0, 2, 0};
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopyPad(bLocal, inputGMB[progress * this->tileDataNum], bCopyParams, padParams);

    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
    inputQueueB.EnQue(bLocal);
}

template <typename T>
__aicore__ inline void SelectV3<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void SelectV3<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<int8_t> bLocal = inputQueueB.DeQue<int8_t>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();

    AscendC::LocalTensor<uint8_t> tmpTensor0 = tmpBuf0.Get<uint8_t>();
    AscendC::LocalTensor<half> tmpTensor1 = tmpBuf1.Get<half>();

    AscendC::Cast(tmpTensor1, bLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
    AscendC::CompareScalar(
        tmpTensor0, tmpTensor1, static_cast<half>(0.0f), AscendC::CMPMODE::NE,
        (this->processDataNum + 255) / 256 * 256);
    AscendC::Select(
        zLocal, tmpTensor0, xLocal, yLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
    inputQueueB.FreeTensor(bLocal);
}

template <typename T>
__aicore__ inline void SelectV3<T>::Process()
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

} // namespace NsSelectV3
#endif // SelectV3_H