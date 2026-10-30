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
 * \file slice_two_dim_small_shape.h
 * \brief
 */
#ifndef SLICE_TWO_DIM_SMALL_SHAPE_H
#define SLICE_TWO_DIM_SMALL_SHAPE_H

#include "slice_base.h"

namespace Slice
{
using namespace AscendC;

template <typename T, typename U, typename V = int8_t>
class SliceTwoDimSmallShape
{
public:
    __aicore__ inline SliceTwoDimSmallShape(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y,
        const SliceTwoDimSmallSapeTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    int64_t blockIdx_ = 0;
    int64_t lastDimBegin_ = 0;
    uint8_t bufferCnt_ = 1;
    const SliceTwoDimSmallSapeTilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
};

template <typename T, typename U, typename V>
__aicore__ inline void SliceTwoDimSmallShape<T, U, V>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y,
                                                      const SliceTwoDimSmallSapeTilingData* tilingData, TPipe* pipeIn)
{
    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    if (tilingData->isBeginConst) {
        lastDimBegin_ = tilingData->lastOneDimOffset;
    } else {
        lastDimBegin_ = static_cast<int64_t>(((__gm__ U*)begin)[1]);
        if constexpr (IsSameType<V, fp4x2_e2m1_t>::value || IsSameType<V, fp4x2_e1m2_t>::value) {
            ascendc_assert(
                !(lastDimBegin_ & 1), "When the input dtype is fp4, the last dimension of offset must be even.\n");
            lastDimBegin_ /= 2;
        }
    }

    // pipeBuffer
    pipe_ = pipeIn;
    pipe_->InitBuffer(vecQue_, bufferCnt_, tilingData_->ubSize);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceTwoDimSmallShape<T, U, V>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    uint32_t blkFactor = tilingData_->blkFactor;
    uint32_t nburst = blockIdx_ < tilingData_->mainCoreNum ? blkFactor : blkFactor - 1;
    uint32_t burstLen = tilingData_->blockLen;
    uint32_t dataOffset = 0;
    if (blockIdx_ < tilingData_->mainCoreNum) {
        // 整块
        nburst = blkFactor;
        dataOffset = blockIdx_ * blkFactor;
    } else {
        // 尾块
        nburst = blkFactor - 1;
        dataOffset = tilingData_->mainCoreNum * blkFactor + (blockIdx_ - tilingData_->mainCoreNum) * (blkFactor - 1);
    }

    uint32_t inputSrcStride = tilingData_->lastOneInputDim * sizeof(T);
    uint32_t inputDstStride = Ops::Base::CeilAlign(burstLen, (uint32_t)BLOCK_SIZE_BYTE);

    DataCopyExtParams copyInParam{static_cast<uint16_t>(nburst), static_cast<uint32_t>(burstLen),
                                  inputSrcStride - burstLen, 0, 0};

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
    DataCopyPad(inputLocal, inputGM_[dataOffset * tilingData_->lastOneInputDim + lastDimBegin_],
                copyInParam, padParams);
    vecQue_.EnQue(inputLocal);
    inputLocal = vecQue_.DeQue<T>();
    uint32_t outBurstLen = tilingData_->lastOneOutputDim * sizeof(T);
    DataCopyExtParams copyOutParams{static_cast<uint16_t>(nburst), static_cast<uint32_t>(outBurstLen),
                                (inputDstStride - outBurstLen) / BLOCK_SIZE_BYTE, 0, 0};
    DataCopyPad(outputGM_[dataOffset * tilingData_->lastOneOutputDim], inputLocal, copyOutParams);
    vecQue_.FreeTensor(inputLocal);
}
}  // namespace Slice

#endif  // SLICE_TWO_DIM_SMALL_SHAPE_H