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
 * \file real_div.h
 * \brief
 */
#ifndef REALDIV_H
#define REALDIV_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "real_div_tiling_data.h"
#include "real_div_tiling_key.h"

namespace MyRealDiv {

using namespace AscendC;

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
class KernelRealDiv {
public:
    __aicore__ inline KernelRealDiv(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const RealDivTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAndCompute(int32_t offset);
    __aicore__ inline void CopyOut(int32_t offset);
    __aicore__ inline void CopyInAndCompute16B(
        LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_Y>& yLocal, int eventIDMTE2ToV,
        int offset);
    __aicore__ inline void CopyInAndComputeInt32(
        LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_Y>& yLocal, int eventIDMTE2ToV,
        int offset);
    __aicore__ inline void CopyInAndComputeBool(
        LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_Y>& yLocal, int eventIDMTE2ToV,
        int offset);

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> castTmp, halfCastTmp;
    GlobalTensor<TYPE_X> x1Gm;
    GlobalTensor<TYPE_X> x2Gm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelRealDiv<TYPE_X, TYPE_Y, BUFFER_NUM>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const RealDivTilingData* tilingData, TPipe* pipeIn)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->pipe = pipeIn;
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
    x1Gm.SetGlobalBuffer((__gm__ TYPE_X*)x1 + globalBufferIndex, this->coreDataNum);
    x2Gm.SetGlobalBuffer((__gm__ TYPE_X*)x2 + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    pipe->InitBuffer(inQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X) * 2);
    pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    if constexpr (IsSameType<TYPE_X, bfloat16_t>::value) {
        pipe->InitBuffer(castTmp, this->tileDataNum * sizeof(float));
    }
    if constexpr (IsSameType<TYPE_X, half>::value) {
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
#else
        pipe->InitBuffer(castTmp, this->tileDataNum * sizeof(float));
#endif
    }
    if constexpr (IsSameType<TYPE_X, bool>::value) {
        pipe->InitBuffer(castTmp, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(halfCastTmp, this->tileDataNum * sizeof(half));
    }
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelRealDiv<TYPE_X, TYPE_Y, BUFFER_NUM>::CopyInAndCompute16B(
    LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_Y>& yLocal, int eventIDMTE2ToV,
    int offset)
{
    LocalTensor<float> x1LocalFp = castTmp.Get<float>();
    LocalTensor<float> x2LocalFp = x1Local.template ReinterpretCast<float>();
    DataCopy(x1Local, x1Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(x1LocalFp, x1Local, RoundMode::CAST_NONE, this->processDataNum);
    DataCopy(x2Local, x2Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(x2LocalFp, x2Local, RoundMode::CAST_NONE, this->processDataNum);
    Div(x1LocalFp, x1LocalFp, x2LocalFp, this->processDataNum);
    Cast(yLocal, x1LocalFp, RoundMode::CAST_RINT, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelRealDiv<TYPE_X, TYPE_Y, BUFFER_NUM>::CopyInAndComputeInt32(
    LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_Y>& yLocal, int eventIDMTE2ToV,
    int offset)
{
    LocalTensor<float> x1LocalFp = x1Local.template ReinterpretCast<float>();
    LocalTensor<float> x2LocalFp = x2Local.template ReinterpretCast<float>();
    DataCopy(x1Local, x1Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(x1LocalFp, x1Local, RoundMode::CAST_NONE, this->processDataNum);
    DataCopy(x2Local, x2Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(x2LocalFp, x2Local, RoundMode::CAST_NONE, this->processDataNum);
    Div(x1LocalFp, x1LocalFp, x2LocalFp, this->processDataNum);
    Cast(yLocal, x1LocalFp, RoundMode::CAST_TRUNC, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelRealDiv<TYPE_X, TYPE_Y, BUFFER_NUM>::CopyInAndComputeBool(
    LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_Y>& yLocal, int eventIDMTE2ToV,
    int offset)
{
    LocalTensor<half> xLocalHalf = halfCastTmp.Get<half>();
    LocalTensor<float> x2LocalFp = castTmp.Get<float>();
    LocalTensor<int8_t> x1LocalInt8 = x1Local.template ReinterpretCast<int8_t>();
    LocalTensor<int8_t> x2LocalInt8 = x2Local.template ReinterpretCast<int8_t>();
    DataCopy(x1Local, x1Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(xLocalHalf, x1LocalInt8, RoundMode::CAST_NONE, this->processDataNum);
    Cast(yLocal, xLocalHalf, RoundMode::CAST_NONE, this->processDataNum);
    DataCopy(x2Local, x2Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(xLocalHalf, x2LocalInt8, RoundMode::CAST_NONE, this->processDataNum);
    Cast(x2LocalFp, xLocalHalf, RoundMode::CAST_NONE, this->processDataNum);
    Div(yLocal, yLocal, x2LocalFp, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelRealDiv<TYPE_X, TYPE_Y, BUFFER_NUM>::CopyInAndCompute(int32_t offset)
{
    LocalTensor<TYPE_X> x1Local = inQueue.template AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> x2Local = x1Local[this->processDataNum];
    LocalTensor<TYPE_Y> yLocal = outQueueY.template AllocTensor<TYPE_Y>();
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    if constexpr (IsSameType<TYPE_X, bfloat16_t>::value) {
        CopyInAndCompute16B(x1Local, x2Local, yLocal, eventIDMTE2ToV, offset);
    } else if constexpr (IsSameType<TYPE_X, int32_t>::value) {
        CopyInAndComputeInt32(x1Local, x2Local, yLocal, eventIDMTE2ToV, offset);
    } else if constexpr (IsSameType<TYPE_X, half>::value) {
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
        DataCopy(x1Local, x1Gm[offset], this->processDataNum);
        DataCopy(x2Local, x2Gm[offset], this->processDataNum);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Div(yLocal, x1Local, x2Local, this->processDataNum);
#else
        CopyInAndCompute16B(x1Local, x2Local, yLocal, eventIDMTE2ToV, offset);
#endif
    } else if constexpr (IsSameType<TYPE_X, bool>::value) {
        CopyInAndComputeBool(x1Local, x2Local, yLocal, eventIDMTE2ToV, offset);
    } else {
        DataCopy(x1Local, x1Gm[offset], this->processDataNum);
        DataCopy(x2Local, x2Gm[offset], this->processDataNum);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Div(yLocal, x1Local, x2Local, this->processDataNum);
    }
    outQueueY.template EnQue<TYPE_Y>(yLocal);
    inQueue.template FreeTensor(x1Local);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelRealDiv<TYPE_X, TYPE_Y, BUFFER_NUM>::CopyOut(int32_t offset)
{
    LocalTensor<TYPE_Y> yLocal = outQueueY.template DeQue<TYPE_Y>();
    DataCopy(yGm[offset], yLocal, this->processDataNum);
    outQueueY.template FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelRealDiv<TYPE_X, TYPE_Y, BUFFER_NUM>::Process()
{
    int32_t loopCount = this->tileNum - 1;
    this->processDataNum = this->tileDataNum;
    int32_t offset = 0;
    for (int32_t i = 0; i < loopCount; i++, offset += this->tileDataNum) {
        CopyInAndCompute(offset);
        CopyOut(offset);
    }
    this->processDataNum = this->tailDataNum;
    CopyInAndCompute(offset);
    CopyOut(offset);
}

} // namespace MyRealDiv
#endif // REALDIV_H