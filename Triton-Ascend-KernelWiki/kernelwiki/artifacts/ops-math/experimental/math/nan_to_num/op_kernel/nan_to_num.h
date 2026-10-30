/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nan_to_num.h
 * \brief
 */
#ifndef NAN_TO_NUM_H_
#define NAN_TO_NUM_H_

#include <math.h>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nan_to_num_tiling_data.h"
#include "nan_to_num_tiling_key.h"

#include "kernel_operator.h"

namespace NsNanToNum {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelNanToNum {
public:
    __aicore__ inline KernelNanToNum(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
        uint64_t tailBlockNum, float nan, float posinf, float neginf);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue, tmpQueueMask, tmpQueue0;

    AscendC::GlobalTensor<TYPE_X> xGm, yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    float nan, posinf, neginf;
};

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::Init(
    GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
    uint64_t tailBlockNum, float nan, float posinf, float neginf)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    this->nan = nan;
    this->posinf = posinf;
    this->neginf = neginf;
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
        pipe.InitBuffer(tmpQueue, this->tileDataNum * sizeof(float));
        // 方案2
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    }
    pipe.InitBuffer(tmpQueueMask, this->tileDataNum * sizeof(uint8_t));
}

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    if constexpr (!std::is_same_v<TYPE_X, bfloat16_t>) {
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask.AllocTensor<uint8_t>();
        AscendC::Compare(mask1Local, xLocal, xLocal, AscendC::CMPMODE::EQ, this->processDataNum);
        AscendC::Select(
            yLocal, mask1Local, xLocal, static_cast<TYPE_X>(this->nan), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            this->processDataNum);
        AscendC::Duplicate(xLocal, static_cast<TYPE_X>(INFINITY), this->processDataNum);
        AscendC::Compare(mask1Local, yLocal, xLocal, AscendC::CMPMODE::EQ, this->processDataNum);
        AscendC::Duplicate(xLocal, static_cast<TYPE_X>(this->posinf), this->processDataNum);
        AscendC::Select(
            yLocal, mask1Local, xLocal, yLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        AscendC::Duplicate(xLocal, static_cast<TYPE_X>(-INFINITY), this->processDataNum);
        AscendC::Compare(mask1Local, yLocal, xLocal, AscendC::CMPMODE::EQ, this->processDataNum);
        AscendC::Duplicate(xLocal, static_cast<TYPE_X>(this->neginf), this->processDataNum);
        AscendC::Select(
            yLocal, mask1Local, xLocal, yLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        outQueueY.EnQue<TYPE_X>(yLocal);
    } else {
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask.AllocTensor<uint8_t>();
        AscendC::LocalTensor<float> tmpLocal = tmpQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::Cast(tmpLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Compare(mask1Local, tmpLocal, tmpLocal, AscendC::CMPMODE::EQ, this->processDataNum);
        AscendC::Select(
            tmpLocal, mask1Local, tmpLocal, static_cast<float>(this->nan), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            this->processDataNum);
        AscendC::Duplicate(tmp0Local, static_cast<float>(INFINITY), this->processDataNum);
        AscendC::Compare(mask1Local, tmpLocal, tmp0Local, AscendC::CMPMODE::EQ, this->processDataNum);
        AscendC::Duplicate(tmp0Local, static_cast<float>(this->posinf), this->processDataNum);
        AscendC::Select(
            tmpLocal, mask1Local, tmp0Local, tmpLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        AscendC::Duplicate(tmp0Local, static_cast<float>(-INFINITY), this->processDataNum);
        AscendC::Compare(mask1Local, tmpLocal, tmp0Local, AscendC::CMPMODE::EQ, this->processDataNum);
        AscendC::Duplicate(tmp0Local, static_cast<float>(this->neginf), this->processDataNum);
        AscendC::Select(
            tmpLocal, mask1Local, tmp0Local, tmpLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        AscendC::Cast(yLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueY.EnQue<TYPE_X>(yLocal);
    }

    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::Process()
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

} // namespace NsNanToNum
#endif // NAN_TO_NUM_H