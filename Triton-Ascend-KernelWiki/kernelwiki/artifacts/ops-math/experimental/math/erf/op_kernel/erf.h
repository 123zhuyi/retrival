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
 * \file erf.h
 * \brief
 */
#ifndef ERF_H
#define ERF_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "erf_tiling_data.h"
#include "erf_tiling_key.h"

namespace MyErf {

using namespace AscendC;

constexpr uint8_t BUFFER_NUM = 2;
constexpr float SCALER_ONE = 1;
constexpr float SCALER_NEGATIVE_ONE = -1;
constexpr float SCALER_P = 0.47047;
constexpr float SCALER_A = 0.3480242;
constexpr float SCALER_B = -0.0958798;
constexpr float SCALER_C = 0.7478556;
constexpr float SCALER_FP16_MAX = 32768;
constexpr float SCALER_FP16_MIN = 1.0 / 32768.0;
constexpr uint32_t SHIFT_BITS = 31;
constexpr int32_t SIGN_FACTOR = -2;
constexpr int32_t SIGN_OFFSET = 1;
constexpr uint32_t UB_OFFSET = 6144;
constexpr uint32_t UB_OFFSET_SMALL = 6912;

template <typename TYPE_X, typename TYPE_Y>
class KernelErf {
public:
    __aicore__ inline KernelErf(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint64_t bigCoreDataNum, uint32_t finalBigTileNum, uint32_t tileDataNum,
        uint32_t bigTailDataNum, uint64_t smallCoreDataNum, uint32_t finalSmallTileNum, uint32_t smallTailDataNum,
        TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t progress);
    __aicore__ inline void CopyOut(uint32_t progress);
    __aicore__ inline void Compute(uint32_t progress);
    __aicore__ inline void Calculate(AscendC::LocalTensor<float>& y, AscendC::LocalTensor<float>& x, uint32_t length);

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf, calcBuf;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelErf<TYPE_X, TYPE_Y>::Init(
    GM_ADDR x, GM_ADDR y, uint64_t bigCoreDataNum, uint32_t finalBigTileNum, uint32_t tileDataNum,
    uint32_t bigTailDataNum, uint64_t smallCoreDataNum, uint32_t finalSmallTileNum, uint32_t smallTailDataNum,
    TPipe* pipeIn)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t coreNum = AscendC::GetBlockNum();
    uint64_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tileDataNum;
    uint64_t coreDataNum;
    if (coreNum != (coreNum - 1)) {
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
        coreDataNum = bigCoreDataNum;
    } else {
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        coreDataNum = smallCoreDataNum;
    }

    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, coreDataNum);

    pipeIn->InitBuffer(inQueueX, BUFFER_NUM, (UB_OFFSET * sizeof(TYPE_X)));
    pipeIn->InitBuffer(outQueueY, BUFFER_NUM, (UB_OFFSET * sizeof(TYPE_X)));
    pipeIn->InitBuffer(calcBuf, 4 * (UB_OFFSET * sizeof(float)));

    if constexpr (!std::is_same_v<TYPE_X, float>) {
        pipeIn->InitBuffer(tmpBuf, 2 * (UB_OFFSET * sizeof(float)));
    }
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelErf<TYPE_X, TYPE_Y>::CopyIn(uint32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelErf<TYPE_X, TYPE_Y>::CopyOut(uint32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelErf<TYPE_X, TYPE_Y>::Compute(uint32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
    if constexpr (!std::is_same_v<TYPE_X, float>) {
        AscendC::LocalTensor<float> p1 = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> p2 = p1[UB_OFFSET];
        AscendC::Cast(p1, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        Calculate(p2, p1, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yLocal, p2, AscendC::RoundMode::CAST_RINT, this->processDataNum);
    } else {
        Calculate(yLocal, xLocal, this->processDataNum);
    }
    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelErf<TYPE_X, TYPE_Y>::Calculate(
    AscendC::LocalTensor<float>& y, AscendC::LocalTensor<float>& x, uint32_t length)
{
    AscendC::LocalTensor<float> tmp = calcBuf.Get<float>();
    AscendC::LocalTensor<float> tmp1 = tmp;
    AscendC::LocalTensor<float> tmp2 = tmp[UB_OFFSET];
    AscendC::LocalTensor<float> tmp3 = tmp[2 * UB_OFFSET];
    AscendC::LocalTensor<float> tmp4 = tmp[3 * UB_OFFSET];

    AscendC::ShiftRight(
        tmp1.ReinterpretCast<uint32_t>(), x.ReinterpretCast<uint32_t>(), (uint32_t)(SHIFT_BITS), length);
    AscendC::Muls(tmp1.ReinterpretCast<int32_t>(), tmp1.ReinterpretCast<int32_t>(), (int32_t)(SIGN_FACTOR), length);
    AscendC::Adds(tmp1.ReinterpretCast<int32_t>(), tmp1.ReinterpretCast<int32_t>(), (int32_t)(SIGN_OFFSET), length);
    AscendC::Cast(tmp1, tmp1.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, length);

    AscendC::Abs(tmp3, x, length);
    AscendC::Duplicate(y, float(SCALER_ONE), length);
    AscendC::Muls(tmp3, tmp3, (float)(SCALER_P), length);
    AscendC::Adds(tmp3, tmp3, (float)(SCALER_ONE), length);
    AscendC::Div(tmp3, y, tmp3, length);

    AscendC::Mul(x, x, x, length);
    AscendC::Muls(x, x, (float)(SCALER_NEGATIVE_ONE), length);
    AscendC::Exp(x, x, length);

    AscendC::Muls(tmp4, tmp3, (float)(SCALER_A), length);
    AscendC::Mul(tmp2, tmp3, tmp3, length);
    AscendC::Axpy(tmp4, tmp2, (float)(SCALER_B), length);
    AscendC::Mul(tmp3, tmp2, tmp3, length);
    AscendC::Axpy(tmp4, tmp3, (float)(SCALER_C), length);
    AscendC::Mul(tmp3, tmp4, x, length);
    AscendC::Axpy(y, tmp3, (float)(SCALER_NEGATIVE_ONE), length);
    AscendC::Mul(y, y, tmp1, length);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelErf<TYPE_X, TYPE_Y>::Process()
{
    uint32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

template <typename TYPE_X, typename TYPE_Y>
class KernelErf_single_core {
public:
    __aicore__ inline KernelErf_single_core(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t tileDataNum, uint32_t bigTailDataNum, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf, calcBuf;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t tileDataNum;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelErf_single_core<TYPE_X, TYPE_Y>::Init(
    GM_ADDR x, GM_ADDR y, uint32_t tileDataNum, uint32_t bigTailDataNum, TPipe* pipeIn)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreId = GetBlockIdx();
    uint32_t coreNum = GetBlockNum();
    uint32_t processDataNum = tileDataNum;
    if (coreId == coreNum - 1) {
        processDataNum = bigTailDataNum;
    }
    if constexpr (std::is_same_v<TYPE_X, float>) {
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + coreId * tileDataNum);
        pipeIn->InitBuffer(inQueueX, 1, UB_OFFSET_SMALL * sizeof(TYPE_X));
        LocalTensor<float> x = inQueueX.AllocTensor<float>();
        DataCopy(x, xGm, processDataNum);
        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID2);

        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + coreId * tileDataNum);
        pipeIn->InitBuffer(outQueueY, 1, UB_OFFSET_SMALL * sizeof(TYPE_Y));
        LocalTensor<float> y = outQueueY.AllocTensor<float>();
        pipeIn->InitBuffer(calcBuf, 5 * (UB_OFFSET_SMALL * sizeof(float)));
        LocalTensor<float> tmp = calcBuf.Get<float>();
        LocalTensor<float> tmp1 = tmp;
        LocalTensor<float> tmp2 = tmp[UB_OFFSET_SMALL];
        LocalTensor<float> tmp3 = tmp[2 * (UB_OFFSET_SMALL)];
        LocalTensor<float> tmp4 = tmp[3 * (UB_OFFSET_SMALL)];
        LocalTensor<float> tmp5 = tmp[4 * (UB_OFFSET_SMALL)];

        Duplicate(y, float(SCALER_ONE), processDataNum);
        WaitFlag<HardEvent::MTE2_V>(eventID2);
        ShiftRight(
            tmp1.ReinterpretCast<uint32_t>(), x.ReinterpretCast<uint32_t>(), (uint32_t)(SHIFT_BITS), processDataNum);
        Muls(tmp1.ReinterpretCast<int32_t>(), tmp1.ReinterpretCast<int32_t>(), (int32_t)(SIGN_FACTOR), processDataNum);
        Adds(tmp1.ReinterpretCast<int32_t>(), tmp1.ReinterpretCast<int32_t>(), (int32_t)(SIGN_OFFSET), processDataNum);
        Cast(tmp1, tmp1.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, processDataNum);

        Abs(tmp2, x, processDataNum);
        Muls(tmp2, tmp2, (float)(SCALER_P), processDataNum);
        Adds(tmp2, tmp2, (float)(SCALER_ONE), processDataNum);
        Div(tmp2, y, tmp2, tileDataNum);

        Mul(tmp5, x, x, processDataNum);
        Muls(tmp5, tmp5, (float)(SCALER_NEGATIVE_ONE), processDataNum);
        Exp(tmp5, tmp5, tileDataNum);

        AscendC::Muls(tmp4, tmp2, (float)(SCALER_A), processDataNum);
        AscendC::Mul(tmp3, tmp2, tmp2, processDataNum);
        AscendC::Axpy(tmp4, tmp3, (float)(SCALER_B), processDataNum);
        AscendC::Mul(tmp2, tmp3, tmp2, processDataNum);
        AscendC::Axpy(tmp4, tmp2, (float)(SCALER_C), processDataNum);
        AscendC::Mul(tmp3, tmp5, tmp4, processDataNum);
        AscendC::Axpy(y, tmp3, (float)(SCALER_NEGATIVE_ONE), processDataNum);
        AscendC::Mul(y, y, tmp1, processDataNum);

        event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId1);
        WaitFlag<HardEvent::V_MTE3>(eventId1);

        DataCopy(yGm, y, processDataNum);
        inQueueX.FreeTensor(x);
        outQueueY.FreeTensor(y);
    }
    if constexpr (!std::is_same_v<TYPE_X, float>) {
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + coreId * tileDataNum);
        pipeIn->InitBuffer(inQueueX, 1, UB_OFFSET_SMALL * sizeof(TYPE_X));
        LocalTensor<TYPE_X> x = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(x, xGm, processDataNum);
        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID2);

        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + coreId * tileDataNum);
        pipeIn->InitBuffer(outQueueY, 1, UB_OFFSET_SMALL * sizeof(TYPE_Y));
        LocalTensor<TYPE_X> y = outQueueY.AllocTensor<TYPE_X>();

        pipeIn->InitBuffer(calcBuf, 6 * (UB_OFFSET_SMALL * sizeof(float)));
        LocalTensor<float> tmp = calcBuf.Get<float>();
        LocalTensor<float> tmp1 = tmp;
        LocalTensor<float> tmp2 = tmp[UB_OFFSET_SMALL];
        LocalTensor<float> tmp3 = tmp[2 * (UB_OFFSET_SMALL)];
        LocalTensor<float> tmp4 = tmp[3 * (UB_OFFSET_SMALL)];
        LocalTensor<float> tmp5 = tmp[4 * (UB_OFFSET_SMALL)];
        LocalTensor<float> tmp6 = tmp[5 * (UB_OFFSET_SMALL)];

        Duplicate(tmp4, float(SCALER_ONE), processDataNum);
        WaitFlag<HardEvent::MTE2_V>(eventID2);
        Cast(tmp2, x, RoundMode::CAST_NONE, processDataNum);
        ShiftRight(
            tmp1.ReinterpretCast<uint32_t>(), tmp2.ReinterpretCast<uint32_t>(), (uint32_t)(SHIFT_BITS), processDataNum);
        Muls(tmp1.ReinterpretCast<int32_t>(), tmp1.ReinterpretCast<int32_t>(), (int32_t)(SIGN_FACTOR), processDataNum);
        Adds(tmp1.ReinterpretCast<int32_t>(), tmp1.ReinterpretCast<int32_t>(), (int32_t)(SIGN_OFFSET), tileDataNum);
        Cast(tmp1, tmp1.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, processDataNum);

        Abs(tmp3, tmp2, processDataNum);
        Muls(tmp3, tmp3, (float)(SCALER_P), processDataNum);
        Adds(tmp3, tmp3, (float)(SCALER_ONE), processDataNum);
        Div(tmp3, tmp4, tmp3, processDataNum);

        Mul(tmp2, tmp2, tmp2, processDataNum);
        Muls(tmp2, tmp2, (float)(SCALER_NEGATIVE_ONE), processDataNum);
        Exp(tmp2, tmp2, processDataNum);

        AscendC::Muls(tmp6, tmp3, (float)(SCALER_A), processDataNum);
        AscendC::Mul(tmp5, tmp3, tmp3, processDataNum);
        AscendC::Axpy(tmp6, tmp5, (float)(SCALER_B), processDataNum);
        AscendC::Mul(tmp3, tmp5, tmp3, processDataNum);
        AscendC::Axpy(tmp6, tmp3, (float)(SCALER_C), processDataNum);
        AscendC::Mul(tmp5, tmp6, tmp2, processDataNum);

        AscendC::Axpy(tmp4, tmp5, (float)(SCALER_NEGATIVE_ONE), processDataNum);
        AscendC::Mul(tmp4, tmp4, tmp1, processDataNum);

        Cast(y, tmp4, RoundMode::CAST_RINT, processDataNum);
        event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId1);
        WaitFlag<HardEvent::V_MTE3>(eventId1);

        DataCopy(yGm, y, processDataNum);
        inQueueX.FreeTensor(x);
        outQueueY.FreeTensor(y);
    }
}

} // namespace MyErf
#endif // ERF_H
