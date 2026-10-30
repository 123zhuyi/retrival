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
 * \file axpy_v2.h
 * \brief
 */
#ifndef __AXPY_V2_H__
#define __AXPY_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "axpy_v2_tiling_data.h"
#include "axpy_v2_tiling_key.h"

namespace NsAxpyV2 {

using namespace AscendC;

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_ALPHA, typename TYPE_Y, uint64_t BUFFER_NUM>
class AxpyV2 {
public:
    __aicore__ inline AxpyV2(){};

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y, const AxpyV2TilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void InitAlphaLocal();
    __aicore__ inline void Process();

private:
    __aicore__ inline void TotalStage(int32_t offset);

private:
    TPipe* pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;
    TBuf<TPosition::VECCALC> alphaBuf;
    LocalTensor<float> alphaLocalFp;
    LocalTensor<int32_t> alphaLocalInt;
    GlobalTensor<TYPE_X1> x1Gm;
    GlobalTensor<TYPE_X2> x2Gm;
    GlobalTensor<TYPE_ALPHA> alphaGm;
    GlobalTensor<TYPE_Y> yGm;
    TYPE_ALPHA alphaValue;
    float alphaValueFp;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_ALPHA, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void AxpyV2<TYPE_X1, TYPE_X2, TYPE_ALPHA, TYPE_Y, BUFFER_NUM>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y, const AxpyV2TilingData* tilingData, TPipe* pipeIn)
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
    x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + globalBufferIndex, this->coreDataNum);
    x2Gm.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + globalBufferIndex, this->coreDataNum);
    alphaGm.SetGlobalBuffer((__gm__ TYPE_ALPHA*)alpha);
    alphaValue = alphaGm.GetValue(0);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    pipe->InitBuffer(queBind, BUFFER_NUM, this->tileDataNum * sizeof(float) * 2);
    pipe->InitBuffer(alphaBuf, this->tileDataNum * sizeof(float));

    InitAlphaLocal();
}
template <typename TYPE_X1, typename TYPE_X2, typename TYPE_ALPHA, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void AxpyV2<TYPE_X1, TYPE_X2, TYPE_ALPHA, TYPE_Y, BUFFER_NUM>::InitAlphaLocal()
{
    if constexpr (
        IsSameType<TYPE_X1, int32_t>::value && IsSameType<TYPE_X2, int32_t>::value &&
        IsSameType<TYPE_ALPHA, int32_t>::value) {
        alphaLocalInt = alphaBuf.Get<int32_t>();
        Duplicate(alphaLocalInt, alphaValue, this->tileDataNum);
    } else {
        alphaLocalFp = alphaBuf.Get<float>();
        if constexpr (IsSameType<TYPE_ALPHA, bfloat16_t>::value) {
            LocalTensor<bfloat16_t> tmp = alphaLocalFp.template ReinterpretCast<bfloat16_t>();
            Duplicate(tmp[this->tileDataNum], alphaValue, this->tileDataNum);
            Cast(alphaLocalFp, tmp[this->tileDataNum], RoundMode::CAST_NONE, this->tileDataNum);
        } else if constexpr (IsSameType<TYPE_ALPHA, half>::value) {
            LocalTensor<half> tmp = alphaLocalFp.template ReinterpretCast<half>();
            Duplicate(tmp[this->tileDataNum], alphaValue, this->tileDataNum);
            Cast(alphaLocalFp, tmp[this->tileDataNum], RoundMode::CAST_NONE, this->tileDataNum);
        } else if constexpr (IsSameType<TYPE_ALPHA, float>::value) {
            Duplicate(alphaLocalFp, alphaValue, this->tileDataNum);
        } else if constexpr (IsSameType<TYPE_ALPHA, int32_t>::value) {
            LocalTensor<int32_t> tmp = alphaLocalFp.template ReinterpretCast<int32_t>();
            Duplicate(tmp, alphaValue, this->tileDataNum);
            Cast(alphaLocalFp, tmp, RoundMode::CAST_NONE, this->tileDataNum);
        }
    }
}

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_ALPHA, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void AxpyV2<TYPE_X1, TYPE_X2, TYPE_ALPHA, TYPE_Y, BUFFER_NUM>::TotalStage(int32_t offset)
{
    LocalTensor<float> x1LocalFp = queBind.template AllocTensor<float>();
    LocalTensor<float> x2LocalFp = x1LocalFp[this->processDataNum];
    LocalTensor<TYPE_X1> x1Local = x1LocalFp.template ReinterpretCast<TYPE_X1>();
    LocalTensor<TYPE_X2> x2Local = x2LocalFp.template ReinterpretCast<TYPE_X2>();
    LocalTensor<TYPE_Y> yLocal = x1LocalFp.template ReinterpretCast<TYPE_Y>();
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    if constexpr (
        IsSameType<TYPE_X1, int32_t>::value && IsSameType<TYPE_X2, int32_t>::value &&
        IsSameType<TYPE_ALPHA, int32_t>::value) {
        DataCopy(x2Local, x2Gm[offset], this->processDataNum);

        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Mul(x2Local, x2Local, alphaLocalInt, this->processDataNum);

        DataCopy(x1Local, x1Gm[offset], this->processDataNum);

        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Add(x1Local, x1Local, x2Local, this->processDataNum);

        if constexpr (IsSameType<TYPE_Y, float>::value) {
            Cast(yLocal, x1Local, RoundMode::CAST_RINT, this->processDataNum);
        }
        if constexpr (IsSameType<TYPE_Y, half>::value || IsSameType<TYPE_Y, bfloat16_t>::value) {
            Cast(x2LocalFp, x1Local, RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, x2LocalFp, RoundMode::CAST_RINT, this->processDataNum);
        }
    } else {
        if constexpr (IsSameType<TYPE_X1, int32_t>::value && IsSameType<TYPE_X2, int32_t>::value) {
            DataCopy(x1Local, x1Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x1LocalFp, x1Local, RoundMode::CAST_NONE, this->processDataNum);

            DataCopy(x2Local, x2Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x2LocalFp, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        }
        if constexpr (IsSameType<TYPE_X1, float>::value && IsSameType<TYPE_X2, float>::value) {
            DataCopy(x2Local, x2Gm[offset], this->processDataNum);
            DataCopy(x1Local, x1Gm[offset], this->processDataNum);
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        }
        if constexpr (
            (IsSameType<TYPE_X1, half>::value || IsSameType<TYPE_X1, bfloat16_t>::value) &&
            (IsSameType<TYPE_X2, half>::value || IsSameType<TYPE_X2, bfloat16_t>::value)) {
            LocalTensor<TYPE_X1> x1LocalLast = x1Local[this->processDataNum];
            LocalTensor<TYPE_X2> x2LocalLast = x2Local[this->processDataNum];
            DataCopy(x1LocalLast, x1Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x1LocalFp, x1LocalLast, RoundMode::CAST_NONE, this->processDataNum);

            DataCopy(x2LocalLast, x2Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x2LocalFp, x2LocalLast, RoundMode::CAST_NONE, this->processDataNum);
        }
        if constexpr (
            (IsSameType<TYPE_X1, half>::value || IsSameType<TYPE_X1, bfloat16_t>::value) &&
            IsSameType<TYPE_X2, int32_t>::value) {
            LocalTensor<TYPE_X1> x1LocalLast = x1Local[this->processDataNum];
            DataCopy(x1LocalLast, x1Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x1LocalFp, x1LocalLast, RoundMode::CAST_NONE, this->processDataNum);

            DataCopy(x2Local, x2Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x2LocalFp, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        }
        if constexpr (
            (IsSameType<TYPE_X1, half>::value || IsSameType<TYPE_X1, bfloat16_t>::value) &&
            IsSameType<TYPE_X2, float>::value) {
            LocalTensor<TYPE_X1> x1LocalLast = x1Local[this->processDataNum];
            DataCopy(x1LocalLast, x1Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x1LocalFp, x1LocalLast, RoundMode::CAST_NONE, this->processDataNum);

            DataCopy(x2Local, x2Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        }
        if constexpr (
            IsSameType<TYPE_X1, float>::value &&
            (IsSameType<TYPE_X2, half>::value || IsSameType<TYPE_X2, bfloat16_t>::value)) {
            LocalTensor<TYPE_X2> x2LocalLast = x2Local[this->processDataNum];
            DataCopy(x2LocalLast, x2Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x2LocalFp, x2LocalLast, RoundMode::CAST_NONE, this->processDataNum);

            DataCopy(x1Local, x1Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        }
        if constexpr (
            IsSameType<TYPE_X1, int32_t>::value &&
            (IsSameType<TYPE_X2, half>::value || IsSameType<TYPE_X2, bfloat16_t>::value)) {
            LocalTensor<TYPE_X2> x2LocalLast = x2Local[this->processDataNum];
            DataCopy(x2LocalLast, x2Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x2LocalFp, x2LocalLast, RoundMode::CAST_NONE, this->processDataNum);

            DataCopy(x1Local, x1Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x1LocalFp, x1Local, RoundMode::CAST_NONE, this->processDataNum);
        }
        if constexpr (IsSameType<TYPE_X1, int32_t>::value && IsSameType<TYPE_X2, float>::value) {
            DataCopy(x1Local, x1Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x1LocalFp, x1Local, RoundMode::CAST_NONE, this->processDataNum);

            DataCopy(x2Local, x2Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        }
        if constexpr (IsSameType<TYPE_X1, float>::value && IsSameType<TYPE_X2, int32_t>::value) {
            DataCopy(x2Local, x2Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(x2LocalFp, x2Local, RoundMode::CAST_NONE, this->processDataNum);

            DataCopy(x1Local, x1Gm[offset], this->processDataNum);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        }
        MulAddDst(x1LocalFp, x2LocalFp, alphaLocalFp, this->processDataNum);

        if constexpr (IsSameType<TYPE_Y, int32_t>::value) {
            Cast(yLocal, x1LocalFp, RoundMode::CAST_TRUNC, this->processDataNum);
        }
        if constexpr (IsSameType<TYPE_Y, half>::value || IsSameType<TYPE_Y, bfloat16_t>::value) {
            Cast(yLocal, x1LocalFp, RoundMode::CAST_RINT, this->processDataNum);
        }
    }

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(yGm[offset], yLocal, this->processDataNum);

    queBind.template FreeTensor(x1LocalFp);
}

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_ALPHA, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void AxpyV2<TYPE_X1, TYPE_X2, TYPE_ALPHA, TYPE_Y, BUFFER_NUM>::Process()
{
    int32_t loopCount = this->tileNum - 1;
    this->processDataNum = this->tileDataNum;
    int32_t offset = 0;
    for (int32_t i = 0; i < loopCount; i++, offset += this->tileDataNum) {
        TotalStage(offset);
    }
    this->processDataNum = this->tailDataNum;
    TotalStage(offset);
}

} // namespace NsAxpyV2
#endif // AXPY_V2_H