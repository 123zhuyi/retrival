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
 * \file ger_v2.h
 * \brief
 */
#ifndef GERV2_H
#define GERV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ger_v2_tiling_data.h"
#include "ger_v2_tiling_key.h"

namespace NsGerV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class GerV2 {
public:
    __aicore__ inline GerV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR A, GM_ADDR z, const GerV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t row, int32_t progress);
    __aicore__ inline void CopyOut(int32_t row, int32_t progress);
    __aicore__ inline void Compute(int32_t row, int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueA;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> inputGMA;
    GlobalTensor<T> outputGMZ;

    int64_t tileDataNum;
    int64_t CoreRows;
    int64_t coreDataNum;
    int64_t CoreColumns;
    int64_t tileNums;
    int64_t tailTileDataNum;
    int64_t processDataNum;
    float alpha;
};

template <typename T>
__aicore__ inline void GerV2<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR A, GM_ADDR z, const GerV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx(); // 正在运行的核号
    uint32_t xGlobalBufferIndex = tilingData->bigCoreRows * AscendC::GetBlockIdx();
    uint32_t AGlobalBufferIndex = tilingData->bigCoreRows * tilingData->n * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    // 大核
    if (coreNum < tilingData->tailRowNum) {
        this->CoreRows = tilingData->bigCoreRows;
        this->coreDataNum = tilingData->bigCoreRows * tilingData->n;
    } else {
        this->CoreRows = tilingData->smallCoreRows;
        this->coreDataNum = tilingData->smallCoreRows * tilingData->n;
        xGlobalBufferIndex -=
            (tilingData->bigCoreRows - tilingData->smallCoreRows) * (AscendC::GetBlockIdx() - tilingData->tailRowNum);
        AGlobalBufferIndex -= ((tilingData->bigCoreRows - tilingData->smallCoreRows) * tilingData->n) *
                              (AscendC::GetBlockIdx() - tilingData->tailRowNum);
    }
    this->CoreColumns = tilingData->n;
    this->tileNums = tilingData->tileNums;
    this->tailTileDataNum = tilingData->tailTileDataNum;
    this->alpha = tilingData->alpha;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + xGlobalBufferIndex, this->CoreRows);
    inputGMY.SetGlobalBuffer((__gm__ T*)y, this->CoreColumns);
    inputGMA.SetGlobalBuffer((__gm__ T*)A + AGlobalBufferIndex, this->coreDataNum);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + AGlobalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->CoreRows * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->CoreColumns * sizeof(T));
    pipe.InitBuffer(inputQueueA, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline void GerV2<T>::CopyIn(int32_t row, int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::LocalTensor<T> ALocal = inputQueueA.AllocTensor<T>();
    AscendC::DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 2, 0};
    AscendC::DataCopyPad(xLocal, inputGMX[row], xCopyParams, padParams);
    AscendC::DataCopyPad(yLocal, inputGMY[progress * this->tileDataNum], yCopyParams, padParams);
    AscendC::DataCopyPad(
        ALocal, inputGMA[this->CoreColumns * row + progress * this->tileDataNum], yCopyParams, padParams);

    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
    inputQueueA.EnQue(ALocal);
}

template <typename T>
__aicore__ inline void GerV2<T>::CopyOut(int32_t row, int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopyExtParams copyExt{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(outputGMZ[this->CoreColumns * row + progress * this->tileDataNum], zLocal, copyExt);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void GerV2<T>::Compute(int32_t row, int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> ALocal = inputQueueA.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();

    T x = xLocal.GetValue(0);
    AscendC::Muls(yLocal, yLocal, static_cast<T>(this->alpha), this->processDataNum);
    AscendC::Muls(yLocal, yLocal, x, this->processDataNum);
    AscendC::Add(zLocal, ALocal, yLocal, this->processDataNum);

    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
    inputQueueA.FreeTensor(ALocal);
}

template <typename T>
__aicore__ inline void GerV2<T>::Process()
{
    for (int32_t i = 0; i < this->CoreRows; i++) {
        int32_t loopCount = this->tileNums;
        this->processDataNum = this->tileDataNum;
        for (int32_t j = 0; j < loopCount; j++) {
            if (j == this->tileNums - 1) {
                this->processDataNum = this->tailTileDataNum;
            }
            CopyIn(i, j);
            Compute(i, j);
            CopyOut(i, j);
        }
    }
}
} // namespace NsGerV2
#endif // GerV2_H