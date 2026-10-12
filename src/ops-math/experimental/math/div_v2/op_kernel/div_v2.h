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
 * \file div_v2.h
 * \brief
 */
#ifndef DIV_V2_H
#define DIV_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "div_v2_tiling_data.h"
#include "div_v2_tiling_key.h"

namespace NsDivV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class DivV2 {
public:
    __aicore__ inline DivV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const DivV2TilingData* tilingData);
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
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T>
__aicore__ inline void DivV2<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const DivV2TilingData* tilingData)
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

    pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
}

template <typename T>
__aicore__ inline void DivV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void DivV2<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void DivV2<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
        AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        // 将输入从bfloat16转换为float
        AscendC::Cast(tmp0, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp1, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        PipeBarrier<PIPE_V>();
        // 执行计算 (使用float类型)
        AscendC::Div(tmp1, tmp0, tmp1, this->processDataNum);
        PipeBarrier<PIPE_V>();
        // 将结果从float转换回bfloat16
        AscendC::Cast(zLocal, tmp1, AscendC::RoundMode::CAST_ROUND, this->processDataNum);
    } else if constexpr (AscendC::Std::is_same<T, int32_t>::value || AscendC::Std::is_same<T, int16_t>::value) {
        AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        // 将输入从int32_t(或int16_t)转换为float
        AscendC::Cast(tmp0, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp1, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        PipeBarrier<PIPE_V>();
        // 执行计算 (使用float类型)
        AscendC::Div(tmp1, tmp0, tmp1, this->processDataNum);
        PipeBarrier<PIPE_V>();
        // 将结果从float转换回int32_t(或int16_t)
        AscendC::Cast(zLocal, tmp1, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
    } else {
        AscendC::Div(zLocal, xLocal, yLocal, this->processDataNum);
    }
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void DivV2<T>::Process()
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

} // namespace NsDivV2
#endif // DivV2_H