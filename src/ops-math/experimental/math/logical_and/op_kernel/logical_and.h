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
 * \file logical_and.h
 * \brief
 */
#ifndef __LOGICAL_AND_H__
#define __LOGICAL_AND_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "logical_and_tiling_data.h"
#include "logical_and_tiling_key.h"

namespace MyLogicalAnd {

using namespace AscendC;

template <uint64_t BUFFER_NUM>
class LogicalAnd {
public:
    __aicore__ inline LogicalAnd(){};

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LogicalAndTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void TotalStage(int32_t offset);

private:
    TPipe* pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;
    TBuf<QuePosition::VECCALC> xCastTmp;
    GlobalTensor<int8_t> x1Gm;
    GlobalTensor<int8_t> x2Gm;
    GlobalTensor<int8_t> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <uint64_t BUFFER_NUM>
__aicore__ inline void LogicalAnd<BUFFER_NUM>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LogicalAndTilingData* tilingData, TPipe* pipeIn)
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
    x1Gm.SetGlobalBuffer((__gm__ int8_t*)x1 + globalBufferIndex, this->coreDataNum);
    x2Gm.SetGlobalBuffer((__gm__ int8_t*)x2 + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ int8_t*)y + globalBufferIndex, this->coreDataNum);
    pipe->InitBuffer(queBind, BUFFER_NUM, this->tileDataNum * sizeof(int8_t) * 2);
    pipe->InitBuffer(xCastTmp, this->tileDataNum * sizeof(half));
}

template <uint64_t BUFFER_NUM>
__aicore__ inline void LogicalAnd<BUFFER_NUM>::TotalStage(int32_t offset)
{
    LocalTensor<int8_t> x1Local = queBind.template AllocTensor<int8_t>();
    LocalTensor<int8_t> x2Local = x1Local[this->processDataNum];
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    LocalTensor<half> x1LocalHalf = xCastTmp.Get<half>();
    LocalTensor<half> x2LocalHalf = x1Local.template ReinterpretCast<half>();
    DataCopy(x1Local, x1Gm[offset], this->processDataNum);

    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(x1LocalHalf, x1Local, RoundMode::CAST_NONE, this->processDataNum);

    DataCopy(x2Local, x2Gm[offset], this->processDataNum);

    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(x2LocalHalf, x2Local, RoundMode::CAST_NONE, this->processDataNum);
    Mul(x1LocalHalf, x1LocalHalf, x2LocalHalf, this->processDataNum);
    Cast(x1Local, x1LocalHalf, RoundMode::CAST_RINT, this->processDataNum);

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(yGm[offset], x1Local, this->processDataNum);

    queBind.template FreeTensor(x1Local);
}

template <uint64_t BUFFER_NUM>
__aicore__ inline void LogicalAnd<BUFFER_NUM>::Process()
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

} // namespace MyLogicalAnd
#endif // LOGICAL_AND_H