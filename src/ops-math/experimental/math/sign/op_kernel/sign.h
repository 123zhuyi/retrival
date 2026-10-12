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
 * \file sign.h
 * \brief
 */
#ifndef SIGN_H
#define SIGN_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sign_tiling_data.h"
#include "sign_tiling_key.h"

namespace NsSign {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t QUEUE_DEPTH = 1;

template <typename TYPE_X>
class KernelSign {
public:
    __aicore__ inline KernelSign(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
        uint64_t tailBlockNuma);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_X> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_X>
__aicore__ inline void KernelSign<TYPE_X>::Init(
    GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
    uint64_t tailBlockNum)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelSign<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSign<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSign<TYPE_X>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
    if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
        AscendC::LocalTensor<float> xFP32 = tmp1.Get<float>();
        AscendC::LocalTensor<float> tmp = tmp2.Get<float>();
        AscendC::Cast(xFP32, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Sign(tmp, xFP32, this->processDataNum);
        AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_RINT, this->processDataNum);
    } else if constexpr (std::is_same_v<TYPE_X, int32_t>) {
        AscendC::Maxs(xLocal, xLocal, static_cast<int32_t>(-1), this->processDataNum);
        AscendC::Mins(yLocal, xLocal, static_cast<int32_t>(1), this->processDataNum);
    } else if constexpr (std::is_same_v<TYPE_X, int16_t>) {
        AscendC::Maxs(xLocal, xLocal, static_cast<int16_t>(-1), this->processDataNum);
        AscendC::Mins(yLocal, xLocal, static_cast<int16_t>(1), this->processDataNum);

    } else {
        AscendC::Sign(yLocal, xLocal, this->processDataNum);
    }
    outQueueY.EnQue<TYPE_X>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSign<TYPE_X>::Process()
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

} // namespace NsSign
#endif // SIGN_H
