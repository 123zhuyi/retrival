/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file floor.h
 * \brief
 */
#ifndef FLOOR_H
#define FLOOR_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "floor_tiling_data.h"
#include "floor_tiling_key.h"

namespace MyFloor {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelFloor {
public:
    __aicore__ inline KernelFloor() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FloorTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_X> yGm;

    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_X>
__aicore__ inline void MyFloor::KernelFloor<TYPE_X>::Init(GM_ADDR x, GM_ADDR y, const FloorTilingData& tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tilingData.bigCoreDataNum * coreId;
    this->tileDataNum = tilingData.tileDataNum;
    uint64_t coreDataNum;

    if (coreId < tilingData.tailBlockNum) {
        coreDataNum = tilingData.bigCoreDataNum;
        this->tileNum = tilingData.finalBigTileNum;
        this->tailDataNum = tilingData.bigTailDataNum;
    } else {
        coreDataNum = tilingData.smallCoreDataNum;
        this->tileNum = tilingData.finalSmallTileNum;
        this->tailDataNum = tilingData.smallTailDataNum;
        globalBufferIndex -=
            (tilingData.bigCoreDataNum - tilingData.smallCoreDataNum) * (coreId - tilingData.tailBlockNum);
    }

    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, coreDataNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));

    if constexpr (!std::is_same_v<TYPE_X, float32_t>) {
        pipe.InitBuffer(tmpBuf, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X>
__aicore__ inline void MyFloor::KernelFloor<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void MyFloor::KernelFloor<TYPE_X>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();

    if constexpr (std::is_same_v<TYPE_X, float>) {
        AscendC::Floor(yLocal, xLocal, this->processDataNum);
    } else {
        AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
        AscendC::Cast(tmpFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Floor(tmpFloat, tmpFloat, this->processDataNum);
        AscendC::Cast(yLocal, tmpFloat, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
    }

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void MyFloor::KernelFloor<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
    if (progress == this->tileNum - 1) {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->tailDataNum * sizeof(TYPE_X)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[progress * this->tileDataNum], yLocal, copyParams);
    } else {
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    }
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X>
__aicore__ inline void MyFloor::KernelFloor<TYPE_X>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;

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
} // namespace MyFloor
#endif // FLOOR_H