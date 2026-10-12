/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liang Yanglin <@liang-yanglin>
 * - Liu Jun <@kbryantttt>
 * - Zhou Jianhua <@LePenseur>
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Li Xing <@li-xingHIT>
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
 * \file transposev.h
 * \brief
 */
#ifndef TRANSPOSEV_H
#define TRANSPOSEV_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "transposev_tiling_data.h"
#include "transposev_tiling_key.h"

namespace NsTransposev {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MAXDIM = 8;

template <typename T, typename U>
class Transposev {
public:
    __aicore__ inline Transposev(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const TransposevTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<U> yGm;
    AscendC::GlobalTensor<T> zGm;

    int64_t coreDataNum;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    int64_t globalBufferIndex;
    int64_t perm[MAXDIM];
    int64_t dims = 8;
    int64_t Stride[MAXDIM];
    int64_t NewStride[MAXDIM];
};

template <typename T, typename U>
__aicore__ inline void Transposev<T, U>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const TransposevTilingData* tilingData)
{
    this->dims = tilingData->dims;
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    uint32_t coreNum = AscendC::GetBlockIdx();
    this->globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        this->globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                                   (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ T*)x, tilingData->totalnumber);
    yGm.SetGlobalBuffer((__gm__ U*)y, this->dims);
    zGm.SetGlobalBuffer((__gm__ T*)z + this->globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueY, BUFFER_NUM, 8 * sizeof(U));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(int32_t));
    pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(int32_t));
    pipe.InitBuffer(tmp5, this->tileDataNum * sizeof(int32_t));
    pipe.InitBuffer(tmp6, this->tileDataNum * sizeof(int32_t));

    AscendC::LocalTensor<U> yLocal = inQueueY.AllocTensor<U>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->dims * sizeof(U)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<U> padParams{true, 0, 2, 0};
    AscendC::DataCopyPad(yLocal, yGm, copyParams, padParams);

    for (int i = 0; i < dims; i++) {
        perm[i] = yLocal.GetValue(i);
    }
    inQueueY.FreeTensor(yLocal);
    Stride[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; --i)
        Stride[i] = Stride[i + 1] * tilingData->shape[i + 1];

    NewStride[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; --i)
        NewStride[i] = NewStride[i + 1] * tilingData->shape[perm[i + 1]];
}

template <typename T, typename U>
__aicore__ inline void Transposev<T, U>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
    outQueueZ.FreeTensor(zLocal);
}

template <typename T, typename U>
__aicore__ inline void Transposev<T, U>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    auto p1 = tmp1.Get<int32_t>();
    auto p2 = tmp2.Get<float>();
    auto p3 = tmp3.Get<float>();
    auto p4 = tmp4.Get<int32_t>();
    auto p5 = tmp5.Get<int32_t>();
    auto p6 = tmp6.Get<int32_t>();
    AscendC::ArithProgression<int32_t>(
        p1, static_cast<int32_t>(this->globalBufferIndex + progress * this->tileDataNum), static_cast<int32_t>(1),
        static_cast<int32_t>(this->processDataNum));

    AscendC::Sub(p6, p6, p6, this->processDataNum);
    for (int j = 0; j < dims; j++) {
        AscendC::Duplicate<float>(p3, static_cast<float>(NewStride[j]), this->processDataNum);
        AscendC::Cast(p2, p1, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
        AscendC::Div(p2, p2, p3, this->processDataNum);
        AscendC::Cast(p4, p2, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
        AscendC::Muls(p5, p4, static_cast<int32_t>(NewStride[j]), this->processDataNum);
        AscendC::Sub(p1, p1, p5, this->processDataNum);
        AscendC::Muls(p4, p4, static_cast<int32_t>(Stride[perm[j]]), this->processDataNum);
        AscendC::Add(p6, p6, p4, this->processDataNum);
    }

    for (int total = 0; total < this->processDataNum; total++) {
        zLocal.SetValue(total, xGm.GetValue(p6.GetValue(total)));
    }
    outQueueZ.EnQue<T>(zLocal);
}

template <typename T, typename U>
__aicore__ inline void Transposev<T, U>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;

    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsTransposev
#endif