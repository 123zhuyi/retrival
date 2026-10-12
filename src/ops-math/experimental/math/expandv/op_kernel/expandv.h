/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
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
 * \file expandv.h
 * \brief
 */
#ifndef __EXPANDV_H__
#define __EXPANDV_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "expandv_tiling_data.h"
#include "expandv_tiling_key.h"

namespace NsExpandv {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MAX_DIM_DEFAULT = 10;

template <typename T>
class Expandv {
public:
    __aicore__ inline Expandv(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ExpandvTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint64_t GetBroadcastIndexEff(uint64_t linearIdx);
    __aicore__ inline void CopyOut(uint64_t progress);
    __aicore__ inline void Compute(uint64_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    uint64_t globalBufferIndex;
    uint64_t in_rank;
    uint64_t out_rank;
    uint64_t inShape[MAX_DIM_DEFAULT];
    uint64_t outShape[MAX_DIM_DEFAULT];
    uint64_t inStride[MAX_DIM_DEFAULT];
    uint64_t outStride[MAX_DIM_DEFAULT];
    uint64_t strideX1[MAX_DIM_DEFAULT];
};

template <typename T>
__aicore__ inline void Expandv<T>::Init(GM_ADDR x, GM_ADDR y, ExpandvTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreIdx = GetBlockIdx();
    this->globalBufferIndex = tilingData->bigCoreDataNum * coreIdx;
    this->tileDataNum = tilingData->tileDataNum;
    this->in_rank = tilingData->in_rank;
    this->out_rank = tilingData->out_rank;
    const uint64_t* inShapeArr = tilingData->inShapeArr;
    const uint64_t* outShapeArr = tilingData->outShapeArr;
    for (int i = 0; i < MAX_DIM_DEFAULT; ++i) {
        this->inShape[i] = tilingData->inShapeArr[i];
        this->outShape[i] = tilingData->outShapeArr[i];
        this->inStride[i] = tilingData->inStrideArr[i];
        this->outStride[i] = tilingData->outStrideArr[i];
    }
    for (int i = 0; i < this->in_rank; ++i) {
        this->strideX1[i] = (inShape[i] == 1) ? 0 : inStride[i];
    }
    if (coreIdx < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum; // （单位：元素数）
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (coreIdx - tilingData->tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y + this->globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline uint64_t Expandv<T>::GetBroadcastIndexEff(uint64_t linearIdx)
{
    uint64_t idxX = 0;
    uint64_t tmp = linearIdx;
    // 从高维对齐输入维度
    for (int i = 0; i < in_rank; ++i) {
        int out_d = out_rank - in_rank + i; // 输出维度对齐输入维度
        uint64_t coord = tmp / outStride[out_d];
        tmp %= outStride[out_d];
        idxX += coord * strideX1[i]; // strideX1[i] = 0 if inShape[i]==1
    }
    return idxX;
}

template <typename T>
__aicore__ inline void Expandv<T>::CopyOut(uint64_t progress)
{
    LocalTensor<T> yLocal = outQueue.DeQue<T>();
    DataCopy(yGm[progress * tileDataNum], yLocal, this->processDataNum);
    outQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Expandv<T>::Compute(uint64_t progress)
{
    LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
    uint64_t baseGlobalIdx = globalBufferIndex + progress * tileDataNum;
    for (uint64_t t = 0; t < this->processDataNum; t++) {
        uint64_t globalIndex = baseGlobalIdx + t;
        uint64_t idxX = GetBroadcastIndexEff(globalIndex);
        yLocal.SetValue(t, xGm.GetValue(idxX));
    }
    outQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Expandv<T>::Process()
{
    uint64_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (uint64_t i = 0; i < loopCount - 1; i++) {
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsExpandv
#endif // EXPANDV_H