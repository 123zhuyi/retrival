/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file acos_grad.h
 * \brief KernelAcosGrad 类实现
 *
 * 数学公式：
 *   x_grad = y_grad * (-1) / sqrt(1 - x^2)
 *
 * 超出定义域（|x| > 1）时：x^2 > 1 → 1-x^2 < 0 → sqrt(负数) = NaN
 *
 * FP16/BF16 路径：Cast -> float32 计算 -> Cast 回原始类型
 * FP32 路径：直接计算
 */

#ifndef ACOS_GRAD_H
#define ACOS_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "acos_grad_tiling_data.h"
#include "acos_grad_tiling_key.h"

namespace NsAcosGrad {

using namespace AscendC;

template <typename T>
class KernelAcosGrad {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t TMP_BUFFER_NUM = 1;

public:
    __aicore__ inline KernelAcosGrad() {}

    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR x, GM_ADDR xGrad, const AcosGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessBlock(uint32_t ubLoop, uint32_t ubTail);
    __aicore__ inline void CopyIn(int64_t progress, uint32_t currentNum);
    __aicore__ inline void Compute(uint32_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, uint32_t currentNum);

private:
    TPipe pipe_;

    TQue<TPosition::VECIN, BUFFER_NUM> inQueueYGrad_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueXGrad_;

    TBuf<TPosition::VECCALC> yGradF32Buf_;
    TBuf<TPosition::VECCALC> xF32Buf_;
    TBuf<TPosition::VECCALC> tmpF32ABuf_;
    TBuf<TPosition::VECCALC> tmpF32BBuf_;

    GlobalTensor<T> yGradGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> xGradGm_;

    uint32_t blockIdx_ = 0;
    uint32_t blockFormer_ = 0;
    uint32_t ubFormer_ = 0;
    int64_t blockOffset_ = 0;
    uint32_t blockLength_ = 0;
};

template <typename T>
__aicore__ inline void KernelAcosGrad<T>::Init(
    GM_ADDR yGrad, GM_ADDR x, GM_ADDR xGrad, const AcosGradTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    blockFormer_ = tilingData->blockFormer;
    ubFormer_ = tilingData->ubFormer;

    uint64_t totalLength = tilingData->totalLength;
    uint64_t start = static_cast<uint64_t>(blockIdx_) * blockFormer_;

    if (start >= totalLength) {
        blockLength_ = 0;
        return;
    }

    uint64_t remaining = totalLength - start;
    blockLength_ = static_cast<uint32_t>((remaining < blockFormer_) ? remaining : blockFormer_);
    blockOffset_ = static_cast<int64_t>(start);

    yGradGm_.SetGlobalBuffer((__gm__ T*)yGrad + blockOffset_, blockLength_);
    xGm_.SetGlobalBuffer((__gm__ T*)x + blockOffset_, blockLength_);
    xGradGm_.SetGlobalBuffer((__gm__ T*)xGrad + blockOffset_, blockLength_);

    pipe_.InitBuffer(inQueueYGrad_, BUFFER_NUM, ubFormer_ * sizeof(T));
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, ubFormer_ * sizeof(T));
    pipe_.InitBuffer(outQueueXGrad_, BUFFER_NUM, ubFormer_ * sizeof(T));

    pipe_.InitBuffer(yGradF32Buf_, ubFormer_ * sizeof(float));
    pipe_.InitBuffer(xF32Buf_, ubFormer_ * sizeof(float));
    pipe_.InitBuffer(tmpF32ABuf_, ubFormer_ * sizeof(float));
    pipe_.InitBuffer(tmpF32BBuf_, ubFormer_ * sizeof(float));
}

template <typename T>
__aicore__ inline void KernelAcosGrad<T>::CopyIn(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> yGradLocal = inQueueYGrad_.template AllocTensor<T>();
    LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, T(0)};

    DataCopyPad(yGradLocal, yGradGm_[progress * static_cast<int64_t>(ubFormer_)], copyParams, padParams);
    DataCopyPad(xLocal, xGm_[progress * static_cast<int64_t>(ubFormer_)], copyParams, padParams);

    inQueueYGrad_.EnQue(yGradLocal);
    inQueueX_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void KernelAcosGrad<T>::Compute(uint32_t currentNum)
{
    LocalTensor<T> yGradLocal = inQueueYGrad_.template DeQue<T>();
    LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    LocalTensor<T> xGradLocal = outQueueXGrad_.template AllocTensor<T>();

    LocalTensor<float> yGradF32 = yGradF32Buf_.Get<float>();
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();
    LocalTensor<float> tmpF32A = tmpF32ABuf_.Get<float>();
    LocalTensor<float> tmpF32B = tmpF32BBuf_.Get<float>();

    Cast(yGradF32, yGradLocal, RoundMode::CAST_NONE, currentNum);
    PipeBarrier<PIPE_V>();
    Cast(xF32, xLocal, RoundMode::CAST_NONE, currentNum);
    PipeBarrier<PIPE_V>();

    Mul(tmpF32A, xF32, xF32, currentNum);
    PipeBarrier<PIPE_V>();

    Muls(tmpF32A, tmpF32A, static_cast<float>(-1.0f), currentNum);
    PipeBarrier<PIPE_V>();

    Adds(tmpF32B, tmpF32A, static_cast<float>(1.0f), currentNum);
    PipeBarrier<PIPE_V>();

    Sqrt(tmpF32B, tmpF32B, currentNum);
    PipeBarrier<PIPE_V>();

    Muls(yGradF32, yGradF32, static_cast<float>(-1.0f), currentNum);
    PipeBarrier<PIPE_V>();

    Div(yGradF32, yGradF32, tmpF32B, currentNum);
    PipeBarrier<PIPE_V>();

    Cast(xGradLocal, yGradF32, RoundMode::CAST_RINT, currentNum);

    outQueueXGrad_.template EnQue<T>(xGradLocal);
    inQueueYGrad_.FreeTensor(yGradLocal);
    inQueueX_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void KernelAcosGrad<T>::CopyOut(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> xGradLocal = outQueueXGrad_.template DeQue<T>();

    int64_t gmOffset = progress * static_cast<int64_t>(ubFormer_);
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(xGradGm_[gmOffset], xGradLocal, copyParams);

    outQueueXGrad_.FreeTensor(xGradLocal);
}

template <typename T>
__aicore__ inline void KernelAcosGrad<T>::ProcessBlock(uint32_t ubLoop, uint32_t ubTail)
{
    for (uint32_t i = 0; i < ubLoop; i++) {
        CopyIn(static_cast<int64_t>(i), ubFormer_);
        Compute(ubFormer_);
        CopyOut(static_cast<int64_t>(i), ubFormer_);
    }
    if (ubTail > 0) {
        CopyIn(static_cast<int64_t>(ubLoop), ubTail);
        Compute(ubTail);
        CopyOut(static_cast<int64_t>(ubLoop), ubTail);
    }
}

template <typename T>
__aicore__ inline void KernelAcosGrad<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    uint32_t ubLoop = blockLength_ / ubFormer_;
    uint32_t ubTail = blockLength_ % ubFormer_;

    ProcessBlock(ubLoop, ubTail);
}

template <>
class KernelAcosGrad<float> {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t TMP_BUFFER_NUM = 1;

public:
    __aicore__ inline KernelAcosGrad() {}

    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR x, GM_ADDR xGrad, const AcosGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessBlock(uint32_t ubLoop, uint32_t ubTail);
    __aicore__ inline void CopyIn(int64_t progress, uint32_t currentNum);
    __aicore__ inline void Compute(uint32_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, uint32_t currentNum);

private:
    TPipe pipe_;

    TQue<TPosition::VECIN, BUFFER_NUM> inQueueYGrad_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueXGrad_;

    TQue<TPosition::VECCALC, TMP_BUFFER_NUM> tmpQueue1_;
    TQue<TPosition::VECCALC, TMP_BUFFER_NUM> tmpQueue2_;

    GlobalTensor<float> yGradGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> xGradGm_;

    uint32_t blockIdx_ = 0;
    uint32_t blockFormer_ = 0;
    uint32_t ubFormer_ = 0;
    int64_t blockOffset_ = 0;
    uint32_t blockLength_ = 0;
};

__aicore__ inline void KernelAcosGrad<float>::Init(
    GM_ADDR yGrad, GM_ADDR x, GM_ADDR xGrad, const AcosGradTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    blockFormer_ = tilingData->blockFormer;
    ubFormer_ = tilingData->ubFormer;

    uint64_t totalLength = tilingData->totalLength;
    uint64_t start = static_cast<uint64_t>(blockIdx_) * blockFormer_;

    if (start >= totalLength) {
        blockLength_ = 0;
        return;
    }

    uint64_t remaining = totalLength - start;
    blockLength_ = static_cast<uint32_t>((remaining < blockFormer_) ? remaining : blockFormer_);
    blockOffset_ = static_cast<int64_t>(start);

    yGradGm_.SetGlobalBuffer((__gm__ float*)yGrad + blockOffset_, blockLength_);
    xGm_.SetGlobalBuffer((__gm__ float*)x + blockOffset_, blockLength_);
    xGradGm_.SetGlobalBuffer((__gm__ float*)xGrad + blockOffset_, blockLength_);

    pipe_.InitBuffer(inQueueYGrad_, BUFFER_NUM, ubFormer_ * sizeof(float));
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, ubFormer_ * sizeof(float));
    pipe_.InitBuffer(outQueueXGrad_, BUFFER_NUM, ubFormer_ * sizeof(float));
    pipe_.InitBuffer(tmpQueue1_, TMP_BUFFER_NUM, ubFormer_ * sizeof(float));
    pipe_.InitBuffer(tmpQueue2_, TMP_BUFFER_NUM, ubFormer_ * sizeof(float));
}

__aicore__ inline void KernelAcosGrad<float>::CopyIn(int64_t progress, uint32_t currentNum)
{
    LocalTensor<float> yGradLocal = inQueueYGrad_.template AllocTensor<float>();
    LocalTensor<float> xLocal = inQueueX_.template AllocTensor<float>();

    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

    DataCopyPad(yGradLocal, yGradGm_[progress * static_cast<int64_t>(ubFormer_)], copyParams, padParams);
    DataCopyPad(xLocal, xGm_[progress * static_cast<int64_t>(ubFormer_)], copyParams, padParams);

    inQueueYGrad_.EnQue(yGradLocal);
    inQueueX_.EnQue(xLocal);
}

__aicore__ inline void KernelAcosGrad<float>::Compute(uint32_t currentNum)
{
    LocalTensor<float> yGradLocal = inQueueYGrad_.template DeQue<float>();
    LocalTensor<float> xLocal = inQueueX_.template DeQue<float>();
    LocalTensor<float> xGradLocal = outQueueXGrad_.template AllocTensor<float>();

    LocalTensor<float> tmpA = tmpQueue1_.template AllocTensor<float>();
    LocalTensor<float> tmpB = tmpQueue2_.template AllocTensor<float>();

    Mul(tmpA, xLocal, xLocal, currentNum);
    PipeBarrier<PIPE_V>();

    Muls(tmpA, tmpA, static_cast<float>(-1.0f), currentNum);
    PipeBarrier<PIPE_V>();

    Adds(tmpB, tmpA, static_cast<float>(1.0f), currentNum);
    PipeBarrier<PIPE_V>();

    Sqrt(tmpB, tmpB, currentNum);
    PipeBarrier<PIPE_V>();

    Muls(tmpA, yGradLocal, static_cast<float>(-1.0f), currentNum);
    PipeBarrier<PIPE_V>();

    Div(xGradLocal, tmpA, tmpB, currentNum);

    outQueueXGrad_.template EnQue<float>(xGradLocal);
    inQueueYGrad_.FreeTensor(yGradLocal);
    inQueueX_.FreeTensor(xLocal);
    tmpQueue1_.FreeTensor(tmpA);
    tmpQueue2_.FreeTensor(tmpB);
}

__aicore__ inline void KernelAcosGrad<float>::CopyOut(int64_t progress, uint32_t currentNum)
{
    LocalTensor<float> xGradLocal = outQueueXGrad_.template DeQue<float>();

    int64_t gmOffset = progress * static_cast<int64_t>(ubFormer_);
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(xGradGm_[gmOffset], xGradLocal, copyParams);

    outQueueXGrad_.FreeTensor(xGradLocal);
}

__aicore__ inline void KernelAcosGrad<float>::ProcessBlock(uint32_t ubLoop, uint32_t ubTail)
{
    for (uint32_t i = 0; i < ubLoop; i++) {
        CopyIn(static_cast<int64_t>(i), ubFormer_);
        Compute(ubFormer_);
        CopyOut(static_cast<int64_t>(i), ubFormer_);
    }
    if (ubTail > 0) {
        CopyIn(static_cast<int64_t>(ubLoop), ubTail);
        Compute(ubTail);
        CopyOut(static_cast<int64_t>(ubLoop), ubTail);
    }
}

__aicore__ inline void KernelAcosGrad<float>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    uint32_t ubLoop = blockLength_ / ubFormer_;
    uint32_t ubTail = blockLength_ % ubFormer_;

    ProcessBlock(ubLoop, ubTail);
}

} // namespace NsAcosGrad

#endif // ACOS_GRAD_H
