/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
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
 * \file pack_v2l.h
 * \brief
 * */
#ifndef PACKV2L_H
#define PACKV2L_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pack_v2_tiling_data.h"
#include "pack_v2_tiling_key.h"

namespace NsPackV2L {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class PackV2L {
public:
    __aicore__ inline PackV2L(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const PackV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;

    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    int32_t total;
    PackV2TilingData tiling;
};

template <typename T>
__aicore__ inline void PackV2L<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const PackV2TilingData* tilingData)
{
    this->tiling = *tilingData;
    uint64_t coreNum = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tiling.bigCoreDataNum * AscendC::GetBlockIdx();
    this->total = tiling.inputDataNum;
    this->ubPartDataNum = tiling.ubPartDataNum;
    if (tiling.tailBlockNum != 0) {
        if (coreNum < tiling.tailBlockNum) {
            this->coreDataNum = tiling.bigCoreDataNum;
            this->tileNum = tiling.bigCoreLoopNum;
            this->tailDataNum = tiling.bigCoreTailDataNum;
        } else {
            this->coreDataNum = tiling.smallCoreDataNum;
            this->tileNum = tiling.smallCoreLoopNum;
            this->tailDataNum = tiling.smallCoreTailDataNum;
            globalBufferIndex -=
                (tiling.bigCoreDataNum - tiling.smallCoreDataNum) * (AscendC::GetBlockIdx() - tiling.tailBlockNum);
        }
    } else {
        this->coreDataNum = tiling.smallCoreDataNum;
        this->tileNum = tiling.smallCoreLoopNum;
        this->tailDataNum = tiling.smallCoreTailDataNum;
        globalBufferIndex = tiling.smallCoreDataNum * AscendC::GetBlockIdx();
    }

    xGm.SetGlobalBuffer((__gm__ float*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ float*)y + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ float*)z + globalBufferIndex * 2, this->coreDataNum * 2);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(float));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(float));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->ubPartDataNum * sizeof(float) * 2);
}

template <typename T>
__aicore__ inline void PackV2L<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();

    AscendC::DataCopy(xLocal, xGm[progress * this->processDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, yGm[progress * this->processDataNum], this->processDataNum);

    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}
template <typename T>
__aicore__ inline void PackV2L<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

    for (int i = 0; i < this->processDataNum; i++) {
        float real = xLocal.GetValue(i);
        float imag = yLocal.GetValue(i);
        zLocal.SetValue(2 * i, real);
        zLocal.SetValue(2 * i + 1, imag);
    }

    outQueueZ.EnQue<float>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}
template <typename T>
__aicore__ inline void PackV2L<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm[progress * this->ubPartDataNum * 2], zLocal, this->processDataNum * 2);
    outQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void PackV2L<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->ubPartDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}
} // namespace NsPackV2L
#endif // PackV2L_H
