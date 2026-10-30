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
 * \file accumulate_nv2.h
 * \brief
 */
#ifndef __ACCUMULATE_NV2_H__
#define __ACCUMULATE_NV2_H__

#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "accumulate_nv2_tiling_data.h"
#include "accumulate_nv2_tiling_key.h"

namespace MyAccumulateNv2 {

using namespace AscendC;

template <typename TYPE_X, typename TYPE_C>
class AccumulateNv2 {
public:
    __aicore__ inline AccumulateNv2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AccumulateNv2TilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void TotalStage(int32_t offset);
    __aicore__ inline void TotalStageN32B(int32_t offset);
    __aicore__ inline void TotalStage32B(int32_t offset);
    __aicore__ inline __gm__ TYPE_X* GetTensorAddr(uint16_t index);

    __aicore__ inline void HandleNumOne32B(
        int32_t offset, LocalTensor<TYPE_X>& yLocal, LocalTensor<TYPE_X>& xLocal, int32_t eventIDMTE2ToV);
    __aicore__ inline void InitFirstTwo32B(
        int32_t offset, LocalTensor<TYPE_X>& yLocal, LocalTensor<TYPE_X>& xLocal, int32_t eventIDMTE2ToV);
    __aicore__ inline void LoadAndAddOne32B(
        int32_t offset, int idx, LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& yLocal, int32_t eventIDVToMTE2,
        int32_t eventIDMTE2ToV);

    __aicore__ inline void HandleNumOneN32B(
        int32_t offset, LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_C>& xCast, LocalTensor<TYPE_C>& yCast,
        int32_t eventIDMTE2ToV);
    __aicore__ inline void InitFirstTwoN32B(
        int32_t offset, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_X>& x3Local, LocalTensor<TYPE_C>& xCast,
        LocalTensor<TYPE_C>& yCast, int32_t eventIDMTE2ToV);
    __aicore__ inline void HandleThirdN32B(
        int32_t offset, LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_C>& xCast, LocalTensor<TYPE_C>& yCast,
        int32_t eventIDMTE2ToV);
    __aicore__ inline void LoadCastAndAddOneN32B(
        int32_t offset, int idx, LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_C>& xCast, LocalTensor<TYPE_C>& yCast,
        int32_t eventIDVToMTE2, int32_t eventIDMTE2ToV);

private:
    TPipe* pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
    TBuf<QuePosition::VECCALC> castTmp;
    GlobalTensor<TYPE_X> x1Gm, x2Gm, xGm;
    GlobalTensor<TYPE_X> yGm;
    GM_ADDR tensorListPtr = nullptr;
    __gm__ uint64_t* tensorPtr;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t globalBufferIndex;
    uint32_t num;
};

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::Init(
    GM_ADDR x, GM_ADDR y, const AccumulateNv2TilingData* tilingData, TPipe* pipeIn)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    this->globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->pipe = pipeIn;
    this->tensorListPtr = x;
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
    this->num = tilingData->num;
    yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    if constexpr (IsSameType<TYPE_X, int8_t>::value || IsSameType<TYPE_X, uint8_t>::value) { // cast to half
        pipe->InitBuffer(queBind, 1, this->tileDataNum * sizeof(TYPE_X) * 3);
        pipe->InitBuffer(castTmp, this->tileDataNum * sizeof(half));
    } else if constexpr (IsSameType<TYPE_X, half>::value) { // cast to fp32
        pipe->InitBuffer(queBind, 1, this->tileDataNum * sizeof(TYPE_X) * 3);
        pipe->InitBuffer(castTmp, this->tileDataNum * sizeof(float));
    } else { // int32 and float32
        pipe->InitBuffer(queBind, 1, this->tileDataNum * sizeof(TYPE_X) * 2);
    }
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::Process()
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

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::TotalStage(int32_t offset)
{
    if constexpr (
        IsSameType<TYPE_X, int8_t>::value || IsSameType<TYPE_X, uint8_t>::value ||
        IsSameType<TYPE_X, half>::value) { // cast to TYPE_C
        TotalStageN32B(offset);
    } else if constexpr (IsSameType<TYPE_X, float>::value || IsSameType<TYPE_X, int32_t>::value) { // int32 and float32
        TotalStage32B(offset);
    }
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline __gm__ TYPE_X* AccumulateNv2<TYPE_X, TYPE_C>::GetTensorAddr(uint16_t index)
{
    return reinterpret_cast<__gm__ TYPE_X*>(*(tensorPtr + index));
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::TotalStage32B(int32_t offset)
{
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t eventIDVToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    LocalTensor<TYPE_X> xLocal = queBind.template AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> yLocal = xLocal[this->processDataNum];

    if (num == 1) {
        HandleNumOne32B(offset, yLocal, xLocal, eventIDMTE2ToV);
    } else {
        InitFirstTwo32B(offset, yLocal, xLocal, eventIDMTE2ToV);
        for (int i = 2; i < num; i++) {
            LoadAndAddOne32B(offset, i, xLocal, yLocal, eventIDVToMTE2, eventIDMTE2ToV);
        }
    }

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(yGm[offset], yLocal, this->processDataNum);

    queBind.template FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::HandleNumOne32B(
    int32_t offset, LocalTensor<TYPE_X>& yLocal, LocalTensor<TYPE_X>& xLocal, int32_t eventIDMTE2ToV)
{
    xGm.SetGlobalBuffer(GetTensorAddr(0) + globalBufferIndex);
    DataCopy(xLocal, xGm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Duplicate(yLocal, (TYPE_X)0, this->processDataNum);
    Add(yLocal, xLocal, yLocal, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::InitFirstTwo32B(
    int32_t offset, LocalTensor<TYPE_X>& yLocal, LocalTensor<TYPE_X>& xLocal, int32_t eventIDMTE2ToV)
{
    x1Gm.SetGlobalBuffer(GetTensorAddr(0) + globalBufferIndex);
    x2Gm.SetGlobalBuffer(GetTensorAddr(1) + globalBufferIndex);
    DataCopy(yLocal, x1Gm[offset], this->processDataNum);
    DataCopy(xLocal, x2Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Add(yLocal, yLocal, xLocal, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::LoadAndAddOne32B(
    int32_t offset, int idx, LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_X>& yLocal, int32_t eventIDVToMTE2,
    int32_t eventIDMTE2ToV)
{
    xGm.SetGlobalBuffer(GetTensorAddr(idx) + globalBufferIndex);
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    DataCopy(xLocal, xGm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Add(yLocal, yLocal, xLocal, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::TotalStageN32B(int32_t offset)
{
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t eventIDVToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    LocalTensor<TYPE_X> x1Local = queBind.template AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> x2Local = x1Local[this->processDataNum];
    LocalTensor<TYPE_X> x3Local = x2Local[this->processDataNum];
    LocalTensor<TYPE_C> xCast = x2Local.template ReinterpretCast<TYPE_C>();
    LocalTensor<TYPE_C> yCast = castTmp.Get<TYPE_C>();

    if (num == 1) {
        HandleNumOneN32B(offset, x1Local, xCast, yCast, eventIDMTE2ToV);
    } else {
        InitFirstTwoN32B(offset, x2Local, x3Local, xCast, yCast, eventIDMTE2ToV);
        if (num > 2) {
            HandleThirdN32B(offset, x1Local, xCast, yCast, eventIDMTE2ToV);
        }
        for (int i = 3; i < num; i++) {
            LoadCastAndAddOneN32B(offset, i, x1Local, xCast, yCast, eventIDVToMTE2, eventIDMTE2ToV);
        }
        Add(yCast, yCast, xCast, this->processDataNum);
    }

    Cast(x1Local, yCast, RoundMode::CAST_NONE, this->processDataNum);
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(yGm[offset], x1Local, this->processDataNum);

    queBind.template FreeTensor(x1Local);
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::HandleNumOneN32B(
    int32_t offset, LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_C>& xCast, LocalTensor<TYPE_C>& yCast,
    int32_t eventIDMTE2ToV)
{
    xGm.SetGlobalBuffer(GetTensorAddr(0) + globalBufferIndex);
    DataCopy(x1Local, xGm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(xCast, x1Local, RoundMode::CAST_NONE, this->processDataNum);
    Duplicate(yCast, (TYPE_C)0.0f, this->processDataNum);
    Add(yCast, xCast, yCast, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::InitFirstTwoN32B(
    int32_t offset, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_X>& x3Local, LocalTensor<TYPE_C>& xCast,
    LocalTensor<TYPE_C>& yCast, int32_t eventIDMTE2ToV)
{
    x1Gm.SetGlobalBuffer(GetTensorAddr(0) + globalBufferIndex);
    x2Gm.SetGlobalBuffer(GetTensorAddr(1) + globalBufferIndex);

    DataCopy(x2Local, x1Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(yCast, x2Local, RoundMode::CAST_NONE, this->processDataNum);
    DataCopy(x3Local, x2Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(xCast, x3Local, RoundMode::CAST_NONE, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::HandleThirdN32B(
    int32_t offset, LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_C>& xCast, LocalTensor<TYPE_C>& yCast,
    int32_t eventIDMTE2ToV)
{
    xGm.SetGlobalBuffer(GetTensorAddr(2) + globalBufferIndex);
    DataCopy(x1Local, xGm[offset], this->processDataNum);
    Add(yCast, yCast, xCast, this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(xCast, x1Local, RoundMode::CAST_NONE, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_C>
__aicore__ inline void AccumulateNv2<TYPE_X, TYPE_C>::LoadCastAndAddOneN32B(
    int32_t offset, int idx, LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_C>& xCast, LocalTensor<TYPE_C>& yCast,
    int32_t eventIDVToMTE2, int32_t eventIDMTE2ToV)
{
    xGm.SetGlobalBuffer(GetTensorAddr(idx) + globalBufferIndex);
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    DataCopy(x1Local, xGm[offset], this->processDataNum);
    Add(yCast, yCast, xCast, this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(xCast, x1Local, RoundMode::CAST_NONE, this->processDataNum);
}

} // namespace MyAccumulateNv2
#endif // ACCUMULATE_NV2_H