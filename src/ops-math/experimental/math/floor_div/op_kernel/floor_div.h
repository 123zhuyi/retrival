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
 * \file floor_div.h
 * \brief
 * */
#ifndef FLOOR_DIV_H
#define FLOOR_DIV_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "floor_div_tiling_data.h"
#include "floor_div_tiling_key.h"

namespace NsFloorDiv {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
template <typename T, typename V>
class FloorDiv {
public:
    __aicore__ inline FloorDiv(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const FloorDivTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    TBuf<QuePosition::VECCALC> tmp0, tmp1, tmp2, tmp3;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T, typename V>
__aicore__ inline void FloorDiv<T, V>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const FloorDivTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    inputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
    if constexpr (!std::is_same_v<T, float>) {
        pipe.InitBuffer(tmp0, this->tileDataNum * sizeof(V));
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(V));
        if constexpr (std::is_same_v<V, half>) {
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
        }
    }
}

template <typename T, typename V>
__aicore__ inline void FloorDiv<T, V>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T, typename V>
__aicore__ inline void FloorDiv<T, V>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T, typename V>
__aicore__ inline void FloorDiv<T, V>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) { // Float32
        AscendC::Div(xLocal, xLocal, yLocal, this->processDataNum);
        AscendC::Floor(zLocal, xLocal, this->processDataNum);
    } else if constexpr (std::is_same_v<DTYPE_X1, int8_t> || std::is_same_v<DTYPE_X1, uint8_t>) { // 处理int8和uint8分支
        AscendC::LocalTensor<V> xHalf = tmp0.Get<V>();                                            // 转换为V类型
        AscendC::LocalTensor<V> yHalf = tmp1.Get<V>();

        AscendC::LocalTensor<float> xFloat = tmp2.Get<float>(); // 转换为float类型
        AscendC::LocalTensor<float> yFloat = tmp3.Get<float>();

        // 转为int8/uint8->half->float
        AscendC::Cast(xHalf, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(yHalf, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

        AscendC::Cast(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(yFloat, yHalf, AscendC::RoundMode::CAST_NONE, this->processDataNum);

        // 进行floorDiv计算
        AscendC::Div(xFloat, xFloat, yFloat, this->processDataNum);
        AscendC::Floor(yFloat, xFloat, this->processDataNum);

        // 转回float->half->int8/uint8
        AscendC::Cast(yHalf, yFloat, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
        AscendC::Cast(zLocal, yHalf, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
    } else {
        AscendC::LocalTensor<V> xFloat = tmp0.Get<V>(); // 转换为V类型，可能为half或float
        AscendC::LocalTensor<V> yFloat = tmp1.Get<V>();

        AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(yFloat, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

        AscendC::Div(xFloat, xFloat, yFloat, this->processDataNum);
        AscendC::Floor(yFloat, xFloat, this->processDataNum);

        AscendC::Cast(zLocal, yFloat, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
    }
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
    outputQueueZ.EnQue(zLocal);
}

template <typename T, typename V>
__aicore__ inline void FloorDiv<T, V>::Process()
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
} // namespace NsFloorDiv
#endif // FloorDiv_H
