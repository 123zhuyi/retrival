/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
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
 * \file onehot.h
 * \brief
 */
#ifndef __ONEHOT_H__
#define __ONEHOT_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "onehot_tiling_data.h"
#include "onehot_tiling_key.h"
namespace NsOnehot {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template <typename T>
class Onehot {
public:
    __aicore__ inline Onehot(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const OnehotTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t j);
    __aicore__ inline void CopyOut(int32_t progress, int32_t j);
    __aicore__ inline void Compute(int32_t progress, int32_t j);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> cmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> oneBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> zBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dBuf;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    int32_t depth;
    int32_t depthAlign;
    uint64_t maskCmp;
    int repeat = 1;
    AscendC::UnaryRepeatParams repeatParams = {1, 1, 8, 8};
};

template <typename T>
__aicore__ inline void Onehot<T>::Init(GM_ADDR x, GM_ADDR z, const OnehotTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->depth = tilingData->depth;
    this->depthAlign = ((this->depth + 8 - 1) / 8) * 8;
    this->maskCmp = depthAlign;
    this->processDataNum = this->tileDataNum;
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
    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex * this->depth, this->coreDataNum * this->depth);
    pipe.InitBuffer(cmpBuf, 8 * depthAlign * sizeof(uint8_t));
    pipe.InitBuffer(oneBuf, depthAlign * sizeof(float));
    pipe.InitBuffer(zBuf, depthAlign * sizeof(float));
    pipe.InitBuffer(dBuf, depthAlign * sizeof(T));
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, depthAlign * sizeof(T));
}

template <typename T>
__aicore__ inline void Onehot<T>::CopyIn(int32_t progress, int32_t j)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Onehot<T>::CopyOut(int32_t progress, int32_t j)
{
    AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->depth * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(zGm[(progress * this->tileDataNum + j) * this->depth], zLocal, copyParams);
    outQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void Onehot<T>::Compute(int32_t progress, int32_t j)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> cmp = cmpBuf.AllocTensor<uint8_t>();
    AscendC::LocalTensor<float> zfloat = zBuf.AllocTensor<float>();
    AscendC::LocalTensor<float> oneLocal = oneBuf.AllocTensor<float>();
    AscendC::LocalTensor<T> dLocal = dBuf.AllocTensor<T>();
    AscendC::Duplicate(oneLocal, 1.0f, depthAlign);
    AscendC::PipeBarrier<PIPE_V>();
    for (int i = 0; i < this->depth; i++) {
        dLocal.SetValue(i, i);
    }
    for (int i = this->depth; i < depthAlign; i++) {
        dLocal.SetValue(i, -1);
    }
    int32_t xVal = xLocal.GetValue(j);
    AscendC::CompareScalar(cmp, dLocal, xVal, AscendC::CMPMODE::EQ, maskCmp, repeat, repeatParams);
    AscendC::LocalTensor<uint16_t> mask_Dup = cmpBuf.AllocTensor<uint16_t>();
    AscendC::Select(zfloat, mask_Dup[0], oneLocal, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->depth);
    AscendC::Cast(zLocal, zfloat, AscendC::RoundMode::CAST_CEIL, this->depth);
    outQueueZ.EnQue<T>(zLocal);
    inQueueX.FreeTensor(xLocal);
    cmpBuf.FreeTensor(cmp);
    oneBuf.FreeTensor(oneLocal);
    zBuf.FreeTensor(zfloat);
    dBuf.FreeTensor(dLocal);
}

template <typename T>
__aicore__ inline void Onehot<T>::Process()
{
    int32_t loopCount = this->tileNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        for (int32_t j = 0; j < this->processDataNum; j++) {
            CopyIn(i, j);
            Compute(i, j);
            CopyOut(i, j);
        }
    }
}

} // namespace NsOnehot
#endif // ONEHOT_H