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
 * \file slice_v3.h
 * \brief
 */
#ifndef __SLICE_V3_H__
#define __SLICE_V3_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "slice_v3_tiling_data.h"
#include "slice_v3_tiling_key.h"

namespace NsSliceV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_DIM = 8;

template <typename T>
class SliceV3 {
public:
    __aicore__ inline SliceV3(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SliceV3TilingData* tiling_data, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t CalculateInputOffset(uint64_t outerIdx);
    __aicore__ inline int64_t CalculateOutputOffset(uint64_t currentOuterIdx);
    __aicore__ inline void CopyIn(int64_t inputOffset, int64_t innerIdx);
    __aicore__ inline void CopyOut(int64_t outputOffset, int64_t innerIdx);
    __aicore__ inline void SyncMTE2ToMTE3();
    __aicore__ inline void SyncMTE3ToMTE2();

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    LocalTensor<T> tmpUB;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    DataCopyExtParams copyParams;
    uint32_t dim;
    uint32_t lastDimSize;
    uint32_t tileDataNum;
    uint64_t totalBlocks;
    uint32_t usedCoreNum;
    int64_t outerLoopNum;
    int64_t tailCoreNum;
    int64_t innerRepTimes;
    int64_t tailNum;
    int64_t begin[MAX_DIM];
    int64_t inputCumShape[MAX_DIM];
    int64_t outputCumShape[MAX_DIM];
};

template <typename T>
__aicore__ inline void SliceV3<T>::Init(GM_ADDR x, GM_ADDR y, const SliceV3TilingData* tiling_data, TPipe* pipe)
{
    this->pipe_ = pipe;
    ;
    this->usedCoreNum = GetBlockNum();
    ASSERT(usedCoreNum != 0 && "block dim can not be zero!");
    this->dim = tiling_data->dim;
    this->lastDimSize = tiling_data->lastDimSize;
    this->tileDataNum = tiling_data->tileDataNum;
    this->outerLoopNum = tiling_data->outerLoopNum;
    this->tailCoreNum = tiling_data->tailCoreNum;

    for (uint32_t i = 0; i < this->dim; ++i) {
        this->begin[i] = tiling_data->begin[i];
        this->inputCumShape[i] = tiling_data->inputCumShape[i];
        this->outputCumShape[i] = tiling_data->outputCumShape[i];
    }

    int64_t numSmallCores = (int64_t)this->usedCoreNum - this->tailCoreNum;
    int64_t bigCoreRows = this->tailCoreNum * (this->outerLoopNum + 1);
    int64_t smallCoreRows = numSmallCores * this->outerLoopNum;
    this->totalBlocks = (uint64_t)(bigCoreRows + smallCoreRows);

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    // 初始化 UB 缓冲区
    uint32_t ubSizePerBuffer = this->tileDataNum * sizeof(T);
    this->pipe_->InitBuffer(inQueue, BUFFER_NUM, ubSizePerBuffer);
    this->pipe_->InitBuffer(outQueue, BUFFER_NUM, ubSizePerBuffer);
    this->pipe_->InitBuffer(tmp1, this->tileDataNum * sizeof(T));

    this->tmpUB = tmp1.Get<T>();

    if (this->lastDimSize == 0) {
        this->innerRepTimes = 0;
        this->tailNum = 0;
    } else {
        this->innerRepTimes = (this->lastDimSize + this->tileDataNum - 1) / this->tileDataNum;
        this->tailNum = this->lastDimSize % this->tileDataNum;
        if (this->tailNum == 0)
            this->tailNum = this->tileDataNum;
    }
}

template <typename T>
__aicore__ inline int64_t SliceV3<T>::CalculateInputOffset(uint64_t outerIdx)
{
    int64_t inputOffset = 0;
    uint64_t tempIdx = outerIdx;
    int64_t outCoord[MAX_DIM] = {0};

    for (int i = 0; i < static_cast<int>(dim) - 1; ++i) {
        outCoord[i] = static_cast<int64_t>(tempIdx / static_cast<uint64_t>(outputCumShape[i + 1]));
        tempIdx %= static_cast<uint64_t>(outputCumShape[i + 1]);
    }

    for (int i = 0; i < static_cast<int>(dim); ++i) {
        int64_t inCoord = (i < static_cast<int>(dim) - 1) ? (outCoord[i] + begin[i]) : begin[i];

        inputOffset += inCoord * inputCumShape[i];
    }

    return inputOffset;
}

template <typename T>
__aicore__ inline int64_t SliceV3<T>::CalculateOutputOffset(uint64_t currentOuterIdx)
{
    return (int64_t)currentOuterIdx * (int64_t)this->lastDimSize;
}

template <typename T>
__aicore__ inline void SliceV3<T>::CopyIn(int64_t inputOffset, int64_t innerIdx)
{
    uint32_t blockLenElements = (innerIdx == this->innerRepTimes - 1) ? this->tailNum : this->tileDataNum;
    uint32_t bytesToCopy = blockLenElements * sizeof(T);

    copyParams.blockLen = bytesToCopy;
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};

    DataCopyPad(this->tmpUB, xGm[inputOffset + innerIdx * tileDataNum], copyParams, padParams);
}

template <typename T>
__aicore__ inline void SliceV3<T>::CopyOut(int64_t outputOffset, int64_t innerIdx)
{
    uint32_t blockLenElements = (innerIdx == this->innerRepTimes - 1) ? this->tailNum : this->tileDataNum;
    uint32_t bytesToCopy = blockLenElements * sizeof(T);

    copyParams.blockLen = bytesToCopy;

    DataCopyPad(yGm[outputOffset + innerIdx * tileDataNum], this->tmpUB, copyParams);
}
template <typename T>
__aicore__ inline void SliceV3<T>::SyncMTE2ToMTE3()
{
    // MTE2->MTE3: 等待从GM到UB的数据拷贝完成
    event_t eventCopyIn = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventCopyIn);
    WaitFlag<HardEvent::MTE2_MTE3>(eventCopyIn);
}

template <typename T>
__aicore__ inline void SliceV3<T>::SyncMTE3ToMTE2()
{
    // MTE3->MTE2: 等待从UB到GM的数据拷贝完成
    event_t eventCopyOut = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventCopyOut);
    WaitFlag<HardEvent::MTE3_MTE2>(eventCopyOut);
}

template <typename T>
__aicore__ inline void SliceV3<T>::Process()
{
    for (uint64_t currentOuterIdx = GetBlockIdx(); currentOuterIdx < totalBlocks; currentOuterIdx += usedCoreNum) {
        int64_t inputOffset = CalculateInputOffset(currentOuterIdx);
        int64_t outputOffset = CalculateOutputOffset(currentOuterIdx);

        for (int64_t innerIdx = 0; innerIdx < innerRepTimes; ++innerIdx) {
            CopyIn(inputOffset, innerIdx);
            SyncMTE2ToMTE3();
            CopyOut(outputOffset, innerIdx);
            SyncMTE3ToMTE2();
        }
    }
}
} // namespace NsSliceV3
#endif