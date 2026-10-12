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
 * \file bitwise_and.h
 * \brief
 */
#ifndef BITWISE_AND_H_
#define BITWISE_AND_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bitwise_and_tiling_data.h"
#include "bitwise_and_tiling_key.h"

#include "kernel_operator.h"

namespace NsBitwiseAnd {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X1>
class KernelBitwiseAnd {
public:
    __aicore__ inline KernelBitwiseAnd(){};

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
        uint64_t tailBlockNum, uint64_t tmpTileDataNum, uint64_t tmpSmallTailDataNum, uint64_t tmpBigTailDataNum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<TYPE_X1> x1Gm, x2Gm, yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    uint64_t tmpTileDataNum, tmpTailDataNum, tmpProcessDataNum;
};

template <typename TYPE_X1>
__aicore__ inline void KernelBitwiseAnd<TYPE_X1>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
    uint64_t tailBlockNum, uint64_t tmpTileDataNum, uint64_t tmpSmallTailDataNum, uint64_t tmpBigTailDataNum)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    this->tmpTileDataNum = tmpTileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
        this->tmpTailDataNum = tmpBigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        this->tmpTailDataNum = tmpSmallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }
    x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + globalBufferIndex, this->coreDataNum);
    x2Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x2 + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_X1*)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X1));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X1));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X1));
}

template <typename TYPE_X1>
__aicore__ inline void KernelBitwiseAnd<TYPE_X1>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.AllocTensor<TYPE_X1>();
    AscendC::LocalTensor<TYPE_X1> x2Local = inQueueX2.AllocTensor<TYPE_X1>();
    AscendC::DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
    inQueueX1.EnQue(x1Local);
    inQueueX2.EnQue(x2Local);
}

template <typename TYPE_X1>
__aicore__ inline void KernelBitwiseAnd<TYPE_X1>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X1> yLocal = outQueueY.DeQue<TYPE_X1>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X1>
__aicore__ inline void KernelBitwiseAnd<TYPE_X1>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
    AscendC::LocalTensor<TYPE_X1> x2Local = inQueueX2.DeQue<TYPE_X1>();
    if constexpr (!std::is_same_v<TYPE_X1, int32_t>) {
        AscendC::LocalTensor<TYPE_X1> yLocal = outQueueY.AllocTensor<TYPE_X1>();
        AscendC::And(yLocal, x1Local, x2Local, this->processDataNum);
        outQueueY.EnQue<TYPE_X1>(yLocal);
    } else {
        AscendC::LocalTensor<int16_t> yLocal = outQueueY.AllocTensor<int16_t>();
        AscendC::And(
            yLocal, x1Local.template ReinterpretCast<int16_t>(), x2Local.template ReinterpretCast<int16_t>(),
            this->tmpProcessDataNum);
        outQueueY.EnQue<int16_t>(yLocal);
    }
    inQueueX1.FreeTensor(x1Local);
    inQueueX2.FreeTensor(x2Local);
}

template <typename TYPE_X1>
__aicore__ inline void KernelBitwiseAnd<TYPE_X1>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    if constexpr (std::is_same_v<TYPE_X1, int32_t>) {
        this->tmpProcessDataNum = this->tmpTileDataNum;
    }
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    if constexpr (std::is_same_v<TYPE_X1, int32_t>) {
        this->tmpProcessDataNum = this->tmpTailDataNum;
    }
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsBitwiseAnd
#endif // BITWISE_AND_H