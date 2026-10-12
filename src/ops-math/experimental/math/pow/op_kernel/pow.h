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
 * \file pow.h
 * \brief
 */
#ifndef POW_H
#define POW_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pow_tiling_data.h"
#include "pow_tiling_key.h"

namespace MyPow {

/*
x > 0: use exp(y*ln(x))
    x < 0: use [-2*(|y|%2)+1]*exp(y*ln|x|)
    x = 0: 0^0=1 & 0^y=0
*/

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y>
class KernelPow {
public:
    __aicore__ inline KernelPow(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR exponent, GM_ADDR y, const PowTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void PowCompute(
        LocalTensor<float>& xLocal, LocalTensor<float>& expLocal, LocalTensor<float>& yLocal,
        LocalTensor<uint8_t>& mask);
    __aicore__ inline void PowComputeUint8(
        LocalTensor<float>& xLocal, LocalTensor<float>& expLocal, LocalTensor<float>& yLocal,
        LocalTensor<uint8_t>& mask);
    __aicore__ inline void ComputeFor16(
        LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& expLocal, LocalTensor<TYPE_Y>& yLocal);
    __aicore__ inline void ComputeForInt32(
        LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& expLocal, LocalTensor<TYPE_Y>& yLocal);
    __aicore__ inline void ComputeForUInt8(
        LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& expLocal, LocalTensor<TYPE_Y>& yLocal);
    __aicore__ inline void ComputeForInt8(
        LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& expLocal, LocalTensor<TYPE_Y>& yLocal);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> absTmp;
    TBuf<QuePosition::VECCALC> xCastTmp;
    TBuf<QuePosition::VECCALC> expCastTmp;
    TBuf<QuePosition::VECCALC> yCastTmp;
    TBuf<QuePosition::VECCALC> tranCastTmp;
    TBuf<QuePosition::VECCALC> maskTmp;
    TBuf<QuePosition::VECCALC> zeroTmp;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_X> expGm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::Init(
    GM_ADDR x, GM_ADDR exponent, GM_ADDR y, const PowTilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (GetBlockIdx() - tilingData->tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
    expGm.SetGlobalBuffer((__gm__ TYPE_X*)exponent + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X) * 2);
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    pipe.InitBuffer(absTmp, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(zeroTmp, this->tileDataNum * sizeof(float));
    if constexpr (IsSameType<TYPE_X, float32_t>::value || IsSameType<TYPE_X, int32_t>::value) {
        pipe.InitBuffer(maskTmp, this->tileDataNum * sizeof(uint8_t));
    } else {
        pipe.InitBuffer(xCastTmp, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(expCastTmp, this->tileDataNum * sizeof(float));

        if constexpr (IsSameType<TYPE_X, uint8_t>::value || IsSameType<TYPE_X, int8_t>::value) {
            pipe.InitBuffer(tranCastTmp, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(yCastTmp, this->tileDataNum * sizeof(float));
        } else {
            pipe.InitBuffer(maskTmp, this->tileDataNum * sizeof(uint8_t));
        }
    }
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::CopyIn(int32_t progress)
{
    LocalTensor<TYPE_X> xLocal = inQueue.AllocTensor<TYPE_X>();
    DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    DataCopy(xLocal[this->processDataNum], expGm[progress * this->tileDataNum], this->processDataNum);
    inQueue.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::CopyOut(int32_t progress)
{
    LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::PowCompute(
    LocalTensor<float>& xLocal, LocalTensor<float>& expLocal, LocalTensor<float>& yLocal, LocalTensor<uint8_t>& mask)
{
    LocalTensor<float> p1 = absTmp.Get<float>();
    LocalTensor<float> zeros = zeroTmp.Get<float>();
    LocalTensor<float>& scalars = zeros;
    LocalTensor<float>& ones = expLocal;
    Duplicate(scalars, 2.0f, this->processDataNum);
    Abs(p1, xLocal, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Ln(p1, p1, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Mul(p1, p1, expLocal, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Exp(yLocal, p1, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Abs(p1, expLocal, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Div(zeros, p1, scalars, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(zeros, zeros, RoundMode::CAST_TRUNC, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Mul(zeros, zeros, scalars, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Sub(p1, p1, zeros, this->processDataNum);

    PipeBarrier<PIPE_V>();
    Muls(p1, p1, -2.0f, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Adds(p1, p1, 1.0f, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Mul(p1, p1, yLocal, this->processDataNum);

    Duplicate(zeros, 0.0f, this->processDataNum);

    CompareScalar(mask, xLocal, 0.f, CMPMODE::LT, this->processDataNum); // x小于0的部分
    PipeBarrier<PIPE_V>();
    Select(yLocal, mask, p1, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

    CompareScalar(mask, expLocal, 0.f, CMPMODE::EQ, this->processDataNum); // exp等于0的部分
    PipeBarrier<PIPE_V>();
    Duplicate(ones, 1.0f, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Select(p1, mask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

    CompareScalar(mask, xLocal, 0.f, CMPMODE::EQ, this->processDataNum); // x等于0的部分
    PipeBarrier<PIPE_V>();
    Select(yLocal, mask, p1, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::PowComputeUint8(
    LocalTensor<float>& xLocal, LocalTensor<float>& expLocal, LocalTensor<float>& yLocal, LocalTensor<uint8_t>& mask)
{
    LocalTensor<float> p1 = absTmp.Get<float>();
    LocalTensor<float> zeros = zeroTmp.Get<float>();
    LocalTensor<float>& ones = expLocal;
    Ln(p1, xLocal, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Mul(p1, p1, expLocal, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Exp(yLocal, p1, this->processDataNum); // p2 = exp(y*len(x))

    Duplicate(zeros, 0.0f, this->processDataNum);
    CompareScalar(mask, expLocal, 0.f, CMPMODE::EQ, this->processDataNum); // exp等于0的部分
    Duplicate(ones, 1.0f, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Select(p1, mask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

    CompareScalar(mask, xLocal, 0.f, CMPMODE::EQ, this->processDataNum); // x等于0的部分
    PipeBarrier<PIPE_V>();
    Select(yLocal, mask, p1, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::ComputeFor16(
    LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& expLocal, LocalTensor<TYPE_Y>& yLocal)
{
    LocalTensor<float> xLocalFp = xCastTmp.Get<float>();
    LocalTensor<float> expLocalFp = expCastTmp.Get<float>();
    LocalTensor<float> yLocalFp = xLocal.template ReinterpretCast<float>();
    LocalTensor<uint8_t> mask = maskTmp.Get<uint8_t>();
    Cast(xLocalFp, xLocal, RoundMode::CAST_NONE, this->processDataNum);
    Cast(expLocalFp, expLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    PowCompute(xLocalFp, expLocalFp, yLocalFp, mask);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::ComputeForInt32(
    LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& expLocal, LocalTensor<TYPE_Y>& yLocal)
{
    LocalTensor<float> xLocalFp = xLocal.template ReinterpretCast<float>();
    LocalTensor<float> expLocalFp = expLocal.template ReinterpretCast<float>();
    LocalTensor<float> yLocalFp = yLocal.template ReinterpretCast<float>();
    LocalTensor<uint8_t> mask = maskTmp.Get<uint8_t>();
    Cast(xLocalFp, xLocal, RoundMode::CAST_NONE, this->processDataNum);
    Cast(expLocalFp, expLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    PowCompute(xLocalFp, expLocalFp, yLocalFp, mask);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::ComputeForInt8(
    LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& expLocal, LocalTensor<TYPE_Y>& yLocal)
{
    LocalTensor<float> xLocalFp = xCastTmp.Get<float>();
    LocalTensor<float> expLocalFp = expCastTmp.Get<float>();
    LocalTensor<float> yLocalFp = yCastTmp.Get<float>();
    LocalTensor<half> tmpLocal = tranCastTmp.Get<half>();
    LocalTensor<uint8_t> mask = xLocal.template ReinterpretCast<uint8_t>();
    Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(xLocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(tmpLocal, expLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(expLocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    PowCompute(xLocalFp, expLocalFp, yLocalFp, mask);
    PipeBarrier<PIPE_V>();
    Cast(tmpLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, tmpLocal, RoundMode::CAST_RINT, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::ComputeForUInt8(
    LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& expLocal, LocalTensor<TYPE_Y>& yLocal)
{
    LocalTensor<float> xLocalFp = xCastTmp.Get<float>();
    LocalTensor<float> expLocalFp = expCastTmp.Get<float>();
    LocalTensor<float> yLocalFp = yCastTmp.Get<float>();
    LocalTensor<half> tmpLocal = tranCastTmp.Get<half>();
    Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(xLocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(tmpLocal, expLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(expLocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    PowComputeUint8(xLocalFp, expLocalFp, yLocalFp, xLocal);
    PipeBarrier<PIPE_V>();
    Cast(tmpLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, tmpLocal, RoundMode::CAST_RINT, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::Compute(int32_t progress)
{
    LocalTensor<TYPE_X> xLocal = inQueue.DeQue<TYPE_X>();
    LocalTensor<TYPE_X> expLocal = xLocal[this->processDataNum];
    LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
    if constexpr (
        IsSameType<TYPE_X, half>::value || IsSameType<TYPE_X, bfloat16_t>::value ||
        IsSameType<TYPE_X, int16_t>::value) {
        ComputeFor16(xLocal, expLocal, yLocal);
    }
    if constexpr (IsSameType<TYPE_X, int32_t>::value) {
        ComputeForInt32(xLocal, expLocal, yLocal);
    }
    if constexpr (IsSameType<TYPE_X, uint8_t>::value) {
        ComputeForUInt8(xLocal, expLocal, yLocal);
    }
    if constexpr (IsSameType<TYPE_X, int8_t>::value) {
        ComputeForInt8(xLocal, expLocal, yLocal);
    }
    if constexpr (IsSameType<TYPE_X, float32_t>::value) {
        LocalTensor<uint8_t> mask = maskTmp.Get<uint8_t>();
        PowCompute(xLocal, expLocal, yLocal, mask);
    }
    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueue.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::Process()
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

} // namespace MyPow
#endif // POW_H