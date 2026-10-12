/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
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
 * \file fill_v2.h
 * \brief
 */
#ifndef __FILL_V2_H__
#define __FILL_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fill_v2_tiling_data.h"
#include "fill_v2_tiling_key.h"

namespace NsFillV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class FillV2 {
public:
    __aicore__ inline FillV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR fill_value, const FillV2TilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    AscendC::GlobalTensor<T> inputGMX;

    T fillValue_ = (T)0;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T>
__aicore__ inline void FillV2<T>::Init(GM_ADDR x, GM_ADDR fill_value, const FillV2TilingData* tilingData, TPipe* pipe)
{
    uint32_t coreId = AscendC::GetBlockIdx();

    AscendC::GlobalTensor<T> valGm;
    valGm.SetGlobalBuffer((__gm__ T*)fill_value, 1);
    this->fillValue_ = valGm.GetValue(0);

    uint32_t globalBufferIndex;
    this->tileDataNum = tilingData->tileDataNum;
    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
        globalBufferIndex = tilingData->bigCoreDataNum * coreId;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        uint32_t bigData = tilingData->bigCoreDataNum;
        uint32_t smallData = tilingData->smallCoreDataNum;
        uint32_t tailBlocks = tilingData->tailBlockNum;
        globalBufferIndex = tailBlocks * bigData + (coreId - tailBlocks) * smallData;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    pipe->InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline void FillV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void FillV2<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::Duplicate<T>(xLocal, this->fillValue_, this->processDataNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void FillV2<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::DataCopy(inputGMX[progress * this->tileDataNum], xLocal, this->processDataNum);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void FillV2<T>::Process()
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

} // namespace NsFillV2
#endif // FILL_V2_H