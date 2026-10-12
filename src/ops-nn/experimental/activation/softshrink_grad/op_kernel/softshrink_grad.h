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
 * \file softshrink_grad.h
 * \brief SoftShrinkGrad 算子 kernel 类定义
 */

#ifndef SOFTSHRINKGRAD_H
#define SOFTSHRINKGRAD_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softshrink_grad_tiling_data.h"
#include "softshrink_grad_tiling_key.h"

#include <type_traits>

namespace NsSoftShrinkGrad {

using namespace AscendC;

template <typename T, int32_t BUFFER_NUM>
class SoftShrinkGrad {
public:
    __aicore__ inline SoftShrinkGrad(){};

    __aicore__ inline void Init(GM_ADDR input_grad, GM_ADDR input_x, GM_ADDR output_y,
                                const SoftShrinkGradTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, uint32_t currentNum);
    __aicore__ inline void Compute(uint32_t currentNum);
    __aicore__ inline void CopyInPad(LocalTensor<T>& local, GlobalTensor<T>& gm, int64_t offset, uint32_t currentNum);

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inputGradQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputXQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> inputXFloatBuf_;
    TBuf<TPosition::VECCALC> gradFloatBuf_;

    GlobalTensor<T> inputGradGm_;
    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outputGm_;

    int64_t blockLength_ = 0;
    int64_t globalOffset_ = 0;
    uint32_t ubLength_ = 0;
    float lambd_ = 0.5f;
};

template <typename T, int32_t BUFFER_NUM>
__aicore__ inline void SoftShrinkGrad<T, BUFFER_NUM>::Init(GM_ADDR input_grad, GM_ADDR input_x, GM_ADDR output_y,
                                                           const SoftShrinkGradTilingData* tilingData, TPipe* pipe)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    globalOffset_ = blockIdx * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - globalOffset_;
    if (remaining <= 0) {
        blockLength_ = 0;
        return;
    }
    if (tilingData->ubFactor <= 0) {
        blockLength_ = 0;
        return;
    }

    blockLength_ = remaining > tilingData->blockFactor ? tilingData->blockFactor : remaining;
    constexpr uint32_t VECTOR_BYTES = 256U;
    constexpr uint32_t ALIGN_ELEMS = VECTOR_BYTES / sizeof(T);
    ubLength_ = (static_cast<uint32_t>(tilingData->ubFactor) + ALIGN_ELEMS - 1U) / ALIGN_ELEMS * ALIGN_ELEMS;
    lambd_ = tilingData->lambd;

    inputGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input_grad) + globalOffset_, blockLength_);
    inputXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input_x) + globalOffset_, blockLength_);
    outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output_y) + globalOffset_, blockLength_);

    pipe->InitBuffer(inputGradQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe->InitBuffer(inputXQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe->InitBuffer(outputQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe->InitBuffer(maskBuf_, ubLength_ * sizeof(uint8_t));

    if constexpr (!std::is_same<T, float>::value) {
        pipe->InitBuffer(inputXFloatBuf_, ubLength_ * sizeof(float));
        pipe->InitBuffer(gradFloatBuf_, ubLength_ * sizeof(float));
    }
}

template <typename T, int32_t BUFFER_NUM>
__aicore__ inline void SoftShrinkGrad<T, BUFFER_NUM>::CopyInPad(LocalTensor<T>& local, GlobalTensor<T>& gm,
                                                                int64_t offset, uint32_t currentNum)
{
    const uint32_t copyBytes = currentNum * sizeof(T);
    const uint32_t alignedBytes = (copyBytes + 31U) & ~31U;
    const uint32_t rightPadding = (alignedBytes - copyBytes) / sizeof(T);
    DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {
        rightPadding != 0U,
        0,
        static_cast<uint8_t>(rightPadding),
        static_cast<T>(0),
    };
    DataCopyPad(local, gm[offset], copyParams, padParams);
}

template <typename T, int32_t BUFFER_NUM>
__aicore__ inline void SoftShrinkGrad<T, BUFFER_NUM>::CopyIn(int64_t offset, uint32_t currentNum)
{
    LocalTensor<T> gradLocal = inputGradQueue_.template AllocTensor<T>();
    LocalTensor<T> xLocal = inputXQueue_.template AllocTensor<T>();
    CopyInPad(gradLocal, inputGradGm_, offset, currentNum);
    CopyInPad(xLocal, inputXGm_, offset, currentNum);
    inputGradQueue_.EnQue(gradLocal);
    inputXQueue_.EnQue(xLocal);
}

template <typename T, int32_t BUFFER_NUM>
__aicore__ inline void SoftShrinkGrad<T, BUFFER_NUM>::Compute(uint32_t currentNum)
{
    constexpr uint32_t VECTOR_BYTES = 256U;
    constexpr uint32_t FLOAT_VECTOR_ELEMS = VECTOR_BYTES / sizeof(float);
    const uint32_t computeNum = (currentNum + FLOAT_VECTOR_ELEMS - 1U) / FLOAT_VECTOR_ELEMS * FLOAT_VECTOR_ELEMS;

    LocalTensor<T> gradLocal = inputGradQueue_.template DeQue<T>();
    LocalTensor<T> xLocal = inputXQueue_.template DeQue<T>();
    LocalTensor<T> outputLocal = outputQueue_.template AllocTensor<T>();
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();

    if constexpr (std::is_same<T, float>::value) {
        Abs(xLocal, xLocal, computeNum);
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xLocal, lambd_, CMPMODE::LE, computeNum);
        PipeBarrier<PIPE_V>();
        Duplicate(xLocal, 0.0f, computeNum);
        PipeBarrier<PIPE_V>();
        Select(outputLocal, maskLocal, xLocal, gradLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<float> xFloat = inputXFloatBuf_.Get<float>();
        LocalTensor<float> gradFloat = gradFloatBuf_.Get<float>();
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, computeNum);
        PipeBarrier<PIPE_V>();
        Cast(gradFloat, gradLocal, RoundMode::CAST_NONE, computeNum);
        PipeBarrier<PIPE_V>();
        Abs(xFloat, xFloat, computeNum);
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, xFloat, lambd_, CMPMODE::LE, computeNum);
        PipeBarrier<PIPE_V>();
        Duplicate(xFloat, 0.0f, computeNum);
        PipeBarrier<PIPE_V>();
        Select(xFloat, maskLocal, xFloat, gradFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
        PipeBarrier<PIPE_V>();
        Cast(outputLocal, xFloat, RoundMode::CAST_RINT, computeNum);
        PipeBarrier<PIPE_V>();
    }

    outputQueue_.EnQue(outputLocal);
    inputGradQueue_.FreeTensor(gradLocal);
    inputXQueue_.FreeTensor(xLocal);
}

template <typename T, int32_t BUFFER_NUM>
__aicore__ inline void SoftShrinkGrad<T, BUFFER_NUM>::CopyOut(int64_t offset, uint32_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueue_.template DeQue<T>();
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm_[offset], outputLocal, copyParams);
    outputQueue_.FreeTensor(outputLocal);
}

template <typename T, int32_t BUFFER_NUM>
__aicore__ inline void SoftShrinkGrad<T, BUFFER_NUM>::Process()
{
    if (blockLength_ <= 0 || ubLength_ == 0) {
        return;
    }

    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const uint32_t currentNum = static_cast<uint32_t>(remaining > ubLength_ ? ubLength_ : remaining);
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsSoftShrinkGrad
#endif // SOFTSHRINKGRAD_H
