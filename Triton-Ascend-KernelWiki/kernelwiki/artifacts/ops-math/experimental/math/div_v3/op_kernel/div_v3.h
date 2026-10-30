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
 * \file div_v3.h
 * \brief DivV3 kernel — division with rounding mode
 *        mode 0: RealDiv   y = x1 / x2
 *        mode 1: TruncDiv  y = trunc(x1 / x2)
 *        mode 2: FloorDiv  y = floor(x1 / x2)
 */
#ifndef DIV_V3_H
#define DIV_V3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "div_v3_tiling_data.h"
#include "div_v3_tiling_key.h"

namespace NsDivV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MODE_REAL_DIV = 0;
constexpr int32_t MODE_TRUNC_DIV = 1;
constexpr int32_t MODE_FLOOR_DIV = 2;

template <typename T>
class DivV3 {
public:
    __aicore__ inline DivV3(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const DivV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

    __aicore__ inline void ComputeFloat(LocalTensor<T>& xLocal, LocalTensor<T>& yLocal, LocalTensor<T>& zLocal);
    __aicore__ inline void ComputeNeedCast(LocalTensor<T>& xLocal, LocalTensor<T>& yLocal, LocalTensor<T>& zLocal);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBufFloor;

    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
    int32_t divMode = 0;
};

template <typename T>
__aicore__ inline void DivV3<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const DivV3TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * blockIdx;
    this->tileDataNum = tilingData->tileDataNum;
    this->divMode = static_cast<int32_t>(tilingData->divMode);

    if (blockIdx < static_cast<uint32_t>(tilingData->tailBlockNum)) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (blockIdx - tilingData->tailBlockNum);
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    inputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));

    if constexpr (!Std::is_same<T, float>::value) {
        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
    }

    if (this->divMode == MODE_FLOOR_DIV) {
        pipe.InitBuffer(tmpBufFloor, this->tileDataNum * sizeof(uint8_t));
    }
}

template <typename T>
__aicore__ inline void DivV3<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * this->tileDataNum], copyParams, padParams);
    DataCopyPad(yLocal, inputGMY[progress * this->tileDataNum], copyParams, padParams);

    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void DivV3<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMZ[progress * this->tileDataNum], zLocal, copyParams);

    outputQueueZ.FreeTensor(zLocal);
}

// float32 path: compute directly without type conversion
template <typename T>
__aicore__ inline void DivV3<T>::ComputeFloat(LocalTensor<T>& xLocal, LocalTensor<T>& yLocal, LocalTensor<T>& zLocal)
{
    Div(zLocal, xLocal, yLocal, this->processDataNum);

    if (this->divMode == MODE_TRUNC_DIV) {
        PipeBarrier<PIPE_V>();
        Trunc(zLocal, zLocal, this->processDataNum);
    } else if (this->divMode == MODE_FLOOR_DIV) {
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> tmpFloor = tmpBufFloor.Get<uint8_t>();
        Floor(zLocal, zLocal, tmpFloor, this->processDataNum);
    }
}

// non-float path: cast to float32, compute, cast back
template <typename T>
__aicore__ inline void DivV3<T>::ComputeNeedCast(LocalTensor<T>& xLocal, LocalTensor<T>& yLocal, LocalTensor<T>& zLocal)
{
    LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
    LocalTensor<float> tmp1 = tmpBuf1.Get<float>();

    Cast(tmp0, xLocal, RoundMode::CAST_NONE, this->processDataNum);
    Cast(tmp1, yLocal, RoundMode::CAST_NONE, this->processDataNum);
    PipeBarrier<PIPE_V>();

    Div(tmp1, tmp0, tmp1, this->processDataNum);

    if (this->divMode == MODE_TRUNC_DIV) {
        PipeBarrier<PIPE_V>();
        Trunc(tmp1, tmp1, this->processDataNum);
    } else if (this->divMode == MODE_FLOOR_DIV) {
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> tmpFloor = tmpBufFloor.Get<uint8_t>();
        Floor(tmp1, tmp1, tmpFloor, this->processDataNum);
    }

    PipeBarrier<PIPE_V>();

    if constexpr (Std::is_same<T, bfloat16_t>::value) {
        Cast(zLocal, tmp1, RoundMode::CAST_RINT, this->processDataNum);
    } else if constexpr (Std::is_same<T, half>::value) {
        Cast(zLocal, tmp1, RoundMode::CAST_NONE, this->processDataNum);
    } else if constexpr (Std::is_same<T, int32_t>::value || Std::is_same<T, int16_t>::value) {
        Cast(zLocal, tmp1, RoundMode::CAST_TRUNC, this->processDataNum);
    } else {
        Cast(zLocal, tmp1, RoundMode::CAST_NONE, this->processDataNum);
    }
}

template <typename T>
__aicore__ inline void DivV3<T>::Compute(int32_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();

    if constexpr (Std::is_same<T, float>::value) {
        ComputeFloat(xLocal, yLocal, zLocal);
    } else {
        ComputeNeedCast(xLocal, yLocal, zLocal);
    }

    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void DivV3<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == loopCount - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsDivV3
#endif // DIV_V3_H
