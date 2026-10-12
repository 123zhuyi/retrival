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
 * \file rsqrt_grad.h
 * \brief
 */
#ifndef RSQRT_GRAD_H_
#define RSQRT_GRAD_H_

#include <math.h>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "rsqrt_grad_tiling_data.h"
#include "rsqrt_grad_tiling_key.h"

#include "kernel_operator.h"

namespace NsRsqrtGrad {

using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t SINGLE_BUFFER_NUM = 1;

template <typename TYPE_Y>
class KernelRsqrtGrad {
public:
    __aicore__ inline KernelRsqrtGrad(){};

    __aicore__ inline void Init(
        GM_ADDR y, GM_ADDR dy, GM_ADDR z, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
        uint64_t tailBlockNum, uint64_t bufferOpen);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueY, inQueueDY;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0, tmpQueue1, tmpQueue2, tmpQueue3, tmpQueue4;

    AscendC::GlobalTensor<TYPE_Y> yGm, dyGm, zGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
};

template <typename TYPE_Y>
__aicore__ inline void KernelRsqrtGrad<TYPE_Y>::Init(
    GM_ADDR y, GM_ADDR dy, GM_ADDR z, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
    uint64_t tailBlockNum, uint64_t bufferOpen)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    uint64_t BUFFER_NUM = DOUBLE_BUFFER_NUM;
    if (bufferOpen == 0) {
        BUFFER_NUM = SINGLE_BUFFER_NUM;
    }
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
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    dyGm.SetGlobalBuffer((__gm__ TYPE_Y*)dy + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ TYPE_Y*)z + globalBufferIndex, this->coreDataNum);
    if constexpr (!std::is_same_v<TYPE_Y, int32_t>) {
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    }
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

    if constexpr (std::is_same_v<TYPE_Y, int8_t>) {
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue3, this->tileDataNum * sizeof(int32_t));
        pipe.InitBuffer(tmpQueue4, this->tileDataNum * sizeof(int32_t));
    } else if constexpr (std::is_same_v<TYPE_Y, half> || std::is_same_v<TYPE_Y, bfloat16_t>) {
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_Y>
__aicore__ inline void KernelRsqrtGrad<TYPE_Y>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
    AscendC::LocalTensor<TYPE_Y> dyLocal = inQueueDY.AllocTensor<TYPE_Y>();
    AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(dyLocal, dyGm[progress * this->tileDataNum], this->processDataNum);
    inQueueY.EnQue(yLocal);
    inQueueDY.EnQue(dyLocal);
}

template <typename TYPE_Y>
__aicore__ inline void KernelRsqrtGrad<TYPE_Y>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.DeQue<TYPE_Y>();
    AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
    outQueueZ.FreeTensor(zLocal);
}

template <typename TYPE_Y>
__aicore__ inline void KernelRsqrtGrad<TYPE_Y>::Compute(int32_t progress)
{
    if constexpr (std::is_same_v<TYPE_Y, float>) {
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> dyLocal = inQueueDY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::Mul(zLocal, yLocal, yLocal, this->processDataNum);
        AscendC::Mul(zLocal, zLocal, yLocal, this->processDataNum);
        AscendC::Mul(zLocal, zLocal, dyLocal, this->processDataNum);
        AscendC::Muls(zLocal, zLocal, static_cast<TYPE_Y>(-0.5), this->processDataNum);
        outQueueZ.EnQue<float>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueDY.FreeTensor(dyLocal);
    } else if constexpr (std::is_same_v<TYPE_Y, int32_t>) {
        AscendC::LocalTensor<int32_t> zLocal = outQueueZ.AllocTensor<int32_t>();
        AscendC::Duplicate(zLocal, static_cast<TYPE_Y>(-0.5), this->processDataNum);
        outQueueZ.EnQue<int32_t>(zLocal);
    } else if constexpr (std::is_same_v<TYPE_Y, half> || std::is_same_v<TYPE_Y, bfloat16_t>) {
        AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
        AscendC::LocalTensor<TYPE_Y> dyLocal = inQueueDY.DeQue<TYPE_Y>();
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.AllocTensor<TYPE_Y>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::Cast(tmp0Local, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(tmp1Local, tmp0Local, tmp0Local, this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, tmp0Local, this->processDataNum);
        AscendC::Cast(tmp0Local, dyLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, tmp0Local, this->processDataNum);
        AscendC::Muls(tmp1Local, tmp1Local, static_cast<float>(-0.5), this->processDataNum);
        AscendC::Cast(zLocal, tmp1Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueZ.EnQue<TYPE_Y>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueDY.FreeTensor(dyLocal);
    } else {
        // int8类型处理分支
        AscendC::LocalTensor<int8_t> yLocal = inQueueY.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> dyLocal = inQueueDY.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> zLocal = outQueueZ.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> tmp0Local = tmpQueue0.AllocTensor<half>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<int32_t> tmp3Local = tmpQueue3.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> tmp4Local = tmpQueue4.AllocTensor<int32_t>();
        AscendC::Cast(tmp0Local, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp1Local, tmp0Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(tmp2Local, tmp1Local, tmp1Local, this->processDataNum);
        AscendC::Mul(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
        AscendC::Cast(tmp0Local, dyLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp1Local, tmp0Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
        AscendC::Muls(tmp2Local, tmp2Local, static_cast<float>(-0.5), this->processDataNum);

        AscendC::Cast(tmp3Local, tmp2Local, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
        AscendC::Duplicate(tmp4Local, static_cast<int32_t>(255), this->processDataNum);
        AscendC::And(tmp3Local, tmp3Local, tmp4Local, this->processDataNum);
        AscendC::Cast(tmp1Local, tmp3Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        // uint8_int8_overflow_proc
        AscendC::Adds(tmp2Local, tmp1Local, static_cast<float>(128.0), this->processDataNum);
        // tensormodint  tmpscalar=1/256=0.00390625
        AscendC::Muls(tmp2Local, tmp2Local, static_cast<float>(0.00390625), this->processDataNum);
        AscendC::Cast(tmp4Local, tmp2Local, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
        AscendC::Duplicate(tmp3Local, static_cast<int32_t>(128), this->processDataNum);
        AscendC::Mul(tmp4Local, tmp4Local, tmp3Local, this->processDataNum);
        AscendC::Cast(tmp2Local, tmp4Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Sub(tmp2Local, tmp1Local, tmp2Local, this->processDataNum);
        // tensormodint end
        AscendC::Adds(tmp2Local, tmp2Local, static_cast<float>(-128.0), this->processDataNum);
        AscendC::Cast(tmp0Local, tmp2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(zLocal, tmp0Local, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
        // uint8_int8_overflow_proc end
        outQueueZ.EnQue<int8_t>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueDY.FreeTensor(dyLocal);
    }
}

template <typename TYPE_Y>
__aicore__ inline void KernelRsqrtGrad<TYPE_Y>::Process()
{
    int32_t loopCount = this->tileNum;
    if constexpr (!std::is_same_v<TYPE_Y, int32_t>) {
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
    } else {
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }
}

} // namespace NsRsqrtGrad
#endif // RSQRT_GRAD_H