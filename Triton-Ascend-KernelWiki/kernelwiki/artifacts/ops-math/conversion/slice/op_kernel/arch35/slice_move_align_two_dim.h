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
 * \file slice_move_align_two_dim.h
 * \brief
 */
#ifndef SLICE_MOVE_ALIGN_TWO_DIM_H
#define SLICE_MOVE_ALIGN_TWO_DIM_H

#include "slice_base.h"

namespace Slice
{
using namespace AscendC;

template <typename T, typename U, typename V = int8_t>
class SliceMoveAlignTwoDim
{
public:
    __aicore__ inline SliceMoveAlignTwoDim(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y,
                                const SliceMoveAlignLast2DimTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockIdx_ = 0;
    int64_t begin_[NUM_TWO] = {0};
    const SliceMoveAlignLast2DimTilingData* tilingData_ = nullptr;
};

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignTwoDim<T, U, V>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                                            GM_ADDR y, const SliceMoveAlignLast2DimTilingData* tilingData,
                                                            TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipeIn;
    tilingData_ = tilingData;

    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);

    for (int32_t i = 0; i < NUM_TWO; i++) {
        if (tilingData->isBeginConst) {
            begin_[i] = tilingData->begin[i];
        } else {
            begin_[i] = static_cast<int64_t>(((__gm__ U*)begin)[i]);
            if constexpr (IsSameType<V, fp4x2_e2m1_t>::value || IsSameType<V, fp4x2_e1m2_t>::value) {
                ascendc_assert(
                    !(begin_[NUM_TWO - 1] & 1),
                    "When the input dtype is fp4, the last dimension of offset must be even.\n");
                begin_[NUM_TWO - 1] /= 2;
            }
        }
    }

    pipe_->InitBuffer(vecQue_, DOUBLE_BUFFER, tilingData->ubSize / DOUBLE_BUFFER);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignTwoDim<T, U, V>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    int64_t ubSplitLoopsNum = ((blockIdx_ == tilingData_->realCoreNum - 1) && (tilingData_->blkTailFactor != 0))
                                  ? tilingData_->blkTailFactor / tilingData_->ubFactor
                                  : tilingData_->blkFactor / tilingData_->ubFactor;

    int64_t rowsOffsetOutput = blockIdx_ * tilingData_->blkFactor;
    int64_t inputGmAddr = begin_[1] + (begin_[0] + rowsOffsetOutput ) * tilingData_->inputSteps[1];
    int64_t outputGmAddr = rowsOffsetOutput * tilingData_->outputShape[1];

    DataCopyExtParams copyInParam{uint16_t(tilingData_->moveAlignParams.blockCount),
                                  uint32_t(tilingData_->moveAlignParams.blockLen),
                                  tilingData_->moveAlignParams.srcStride - tilingData_->moveAlignParams.blockLen,
                                  tilingData_->moveAlignParams.dstStride, 0};
    DataCopyExtParams copyOutParamsMain{
        1u, uint32_t(tilingData_->moveAlignParams.blockCount * tilingData_->moveAlignParams.blockLen), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (int64_t loops = 0; loops < ubSplitLoopsNum; loops++) {
        LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Compact>(inputLocal, inputGM_[inputGmAddr + loops * tilingData_->ubInLoopSteps],
                                             copyInParam, padParams);

        vecQue_.EnQue(inputLocal);
        inputLocal = vecQue_.DeQue<T>();

        DataCopyPad(outputGM_[outputGmAddr + loops * tilingData_->ubOutLoopSteps], inputLocal, copyOutParamsMain);
        vecQue_.FreeTensor(inputLocal);
    }

    int64_t ubTailFactor_ = ((blockIdx_ == tilingData_->realCoreNum - 1) && (tilingData_->blkTailFactor != 0))
                                ? tilingData_->ubTailTailFactor
                                : tilingData_->ubTailFactor;
    if (ubTailFactor_ > 0) {
        copyInParam.blockCount = ubTailFactor_;
        LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Compact>(
            inputLocal, inputGM_[inputGmAddr + ubSplitLoopsNum * tilingData_->ubInLoopSteps], copyInParam, padParams);
        vecQue_.EnQue(inputLocal);
        inputLocal = vecQue_.DeQue<T>();

        DataCopyExtParams copyOutParamsTail{1, uint32_t(ubTailFactor_ * tilingData_->moveAlignParams.blockLen), 0, 0,
                                            0};
        DataCopyPad(outputGM_[outputGmAddr + ubSplitLoopsNum * tilingData_->ubOutLoopSteps], inputLocal,
                    copyOutParamsTail);
        vecQue_.FreeTensor(inputLocal);
    }
}
}  // namespace Slice

#endif  // SLICE_MOVE_ALIGN_TWO_DIM_H