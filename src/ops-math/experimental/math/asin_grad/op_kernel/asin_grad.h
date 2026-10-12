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
 * \file asin_grad.h
 * \brief AsinGrad kernel implementation.
 *
 * Math: z = dy / sqrt(1 - y * y)
 *
 * FP32 path : compute directly on float and reuse input UB for the denominator.
 * FP16 path : compute directly on half.
 * BF16 path : Cast -> float32 compute -> Cast back to BF16.
 */

#ifndef EXPERIMENTAL_MATH_ASIN_GRAD_H_
#define EXPERIMENTAL_MATH_ASIN_GRAD_H_

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "asin_grad_tiling_data.h"
#include "asin_grad_tiling_key.h"

namespace NsAsinGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t DATA_COPY_ALIGN_BYTES = 32;

__aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

__aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
{
    if (align == 0) {
        return value;
    }
    return CeilDiv(value, align) * align;
}

template <typename Kernel>
__aicore__ inline void ProcessTiles(Kernel& kernel, int64_t blockLength, int64_t tileLength)
{
    if (tileLength == 0) {
        return;
    }
    int64_t tileNum = CeilDiv(blockLength, tileLength);
    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t validLength = tileLength;
        if (i == tileNum - 1) {
            validLength = blockLength - i * tileLength;
        }
        kernel.CopyIn(i, validLength);
        kernel.Compute(validLength);
        kernel.CopyOut(i, validLength);
    }
}

template <typename Derived, typename T>
class AsinGradKernelBase {
public:
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z, const AsinGradTilingData* tilingData)
    {
        InitBlockGm(y, dy, z, tilingData);
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(inQueueDy_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(T));
        static_cast<Derived&>(*this).InitExtraBuffers();
    }

    __aicore__ inline void Process() { ProcessTiles(static_cast<Derived&>(*this), blockLength_, tileLength_); }

    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        int64_t alignElements = CeilDiv(DATA_COPY_ALIGN_BYTES, static_cast<int64_t>(sizeof(T)));
        int64_t alignedLength = AlignUp(curTileLength, alignElements);
        LocalTensor<T> yLocal = inQueueY_.template AllocTensor<T>();
        LocalTensor<T> dyLocal = inQueueDy_.template AllocTensor<T>();
        if (alignedLength == curTileLength) {
            DataCopy(yLocal, yGm_[progress * tileLength_], static_cast<uint32_t>(curTileLength));
            DataCopy(dyLocal, dyGm_[progress * tileLength_], static_cast<uint32_t>(curTileLength));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedLength - curTileLength),
                                              static_cast<T>(0)};
            DataCopyPad(yLocal, yGm_[progress * tileLength_], copyParams, padParams);
            DataCopyPad(dyLocal, dyGm_[progress * tileLength_], copyParams, padParams);
        }
        inQueueY_.EnQue(yLocal);
        inQueueDy_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        LocalTensor<T> zLocal = outQueueZ_.template DeQue<T>();
        int64_t alignElements = CeilDiv(DATA_COPY_ALIGN_BYTES, static_cast<int64_t>(sizeof(T)));
        int64_t alignedLength = AlignUp(curTileLength, alignElements);
        if (alignedLength == curTileLength) {
            DataCopy(zGm_[progress * tileLength_], zLocal, static_cast<uint32_t>(curTileLength));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(zGm_[progress * tileLength_], zLocal, copyParams);
        }
        outQueueZ_.FreeTensor(zLocal);
    }

protected:
    __aicore__ inline int64_t GetComputeLength(int64_t validLength)
    {
        int64_t typeAlignElements = CeilDiv(DATA_COPY_ALIGN_BYTES, static_cast<int64_t>(sizeof(T)));
        int64_t fp32AlignElements = CeilDiv(DATA_COPY_ALIGN_BYTES, static_cast<int64_t>(sizeof(float)));
        int64_t alignElements = typeAlignElements > fp32AlignElements ? typeAlignElements : fp32AlignElements;
        return AlignUp(validLength, alignElements);
    }

    __aicore__ inline void InitBlockGm(GM_ADDR y, GM_ADDR dy, GM_ADDR z, const AsinGradTilingData* tilingData)
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t offset = tilingData->formerLength * tilingData->formerNum;
        blockLength_ = tilingData->tailLength;
        if (blockIdx < tilingData->formerNum) {
            blockLength_ = tilingData->formerLength;
            offset = tilingData->formerLength * blockIdx;
        }
        yGm_.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
        dyGm_.SetGlobalBuffer((__gm__ T*)dy + offset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ T*)z + offset, blockLength_);
        tileLength_ = tilingData->tileLength;
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueDy_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> dyGm_;
    GlobalTensor<T> zGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

// FP32 path: compute directly on float and reuse yLocal for the denominator.
template <typename T>
class KernelAsinGradFp32 : public AsinGradKernelBase<KernelAsinGradFp32<T>, T> {
public:
    __aicore__ inline KernelAsinGradFp32() {}

    __aicore__ inline void InitExtraBuffers() {}

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<T> yLocal = this->inQueueY_.template DeQue<T>();
        LocalTensor<T> dyLocal = this->inQueueDy_.template DeQue<T>();
        LocalTensor<T> zLocal = this->outQueueZ_.template AllocTensor<T>();
        int64_t n = this->GetComputeLength(curTileLength);
        Mul(yLocal, yLocal, yLocal, static_cast<uint32_t>(n)); // y * y
        PipeBarrier<PIPE_V>();
        Muls(yLocal, yLocal, static_cast<T>(-1.0f), static_cast<uint32_t>(n)); // -y * y
        PipeBarrier<PIPE_V>();
        Adds(yLocal, yLocal, static_cast<T>(1.0f), static_cast<uint32_t>(n)); // 1 - y * y
        PipeBarrier<PIPE_V>();
        Sqrt(yLocal, yLocal, static_cast<uint32_t>(n)); // sqrt(1 - y*y)
        PipeBarrier<PIPE_V>();
        Div(zLocal, dyLocal, yLocal, static_cast<uint32_t>(n)); // dy / sqrt(1 - y*y)
        this->outQueueZ_.template EnQue<T>(zLocal);
        this->inQueueY_.FreeTensor(yLocal);
        this->inQueueDy_.FreeTensor(dyLocal);
    }
};

// FP16 computes directly on half; BF16 upcasts to float32, computes, then downcasts back.
template <typename T>
class KernelAsinGradCast : public AsinGradKernelBase<KernelAsinGradCast<T>, T> {
public:
    __aicore__ inline KernelAsinGradCast() {}

    __aicore__ inline void InitExtraBuffers()
    {
        if constexpr (!std::is_same_v<T, half>) {
            this->pipe_.InitBuffer(yFp32Buf_, this->tileLength_ * sizeof(float));
            this->pipe_.InitBuffer(dyFp32Buf_, this->tileLength_ * sizeof(float));
            this->pipe_.InitBuffer(zFp32Buf_, this->tileLength_ * sizeof(float));
        }
    }

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<T> yLocal = this->inQueueY_.template DeQue<T>();
        LocalTensor<T> dyLocal = this->inQueueDy_.template DeQue<T>();
        LocalTensor<T> zLocal = this->outQueueZ_.template AllocTensor<T>();
        int64_t n = this->GetComputeLength(curTileLength);
        if constexpr (std::is_same_v<T, half>) {
            Mul(zLocal, yLocal, yLocal, static_cast<uint32_t>(n)); // y * y
            PipeBarrier<PIPE_V>();
            Muls(zLocal, zLocal, static_cast<T>(-1.0f), static_cast<uint32_t>(n)); // -y * y
            PipeBarrier<PIPE_V>();
            Adds(zLocal, zLocal, static_cast<T>(1.0f), static_cast<uint32_t>(n)); // 1 - y * y
            PipeBarrier<PIPE_V>();
            Sqrt(zLocal, zLocal, static_cast<uint32_t>(n)); // sqrt(1 - y*y)
            PipeBarrier<PIPE_V>();
            Div(zLocal, dyLocal, zLocal, static_cast<uint32_t>(n)); // dy / sqrt(1 - y*y)
        } else {
            LocalTensor<float> yFp32 = yFp32Buf_.template Get<float>();
            LocalTensor<float> dyFp32 = dyFp32Buf_.template Get<float>();
            LocalTensor<float> zFp32 = zFp32Buf_.template Get<float>();
            Cast(yFp32, yLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(n));
            Cast(dyFp32, dyLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(n));
            PipeBarrier<PIPE_V>();
            Mul(yFp32, yFp32, yFp32, static_cast<uint32_t>(n)); // y * y
            PipeBarrier<PIPE_V>();
            Muls(yFp32, yFp32, -1.0f, static_cast<uint32_t>(n)); // -y * y
            PipeBarrier<PIPE_V>();
            Adds(yFp32, yFp32, 1.0f, static_cast<uint32_t>(n)); // 1 - y * y
            PipeBarrier<PIPE_V>();
            Sqrt(yFp32, yFp32, static_cast<uint32_t>(n)); // sqrt(1 - y*y)
            PipeBarrier<PIPE_V>();
            Div(zFp32, dyFp32, yFp32, static_cast<uint32_t>(n)); // dy / sqrt(1 - y*y)
            PipeBarrier<PIPE_V>();
            Cast(zLocal, zFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(n));
        }
        this->outQueueZ_.template EnQue<T>(zLocal);
        this->inQueueY_.FreeTensor(yLocal);
        this->inQueueDy_.FreeTensor(dyLocal);
    }

private:
    TBuf<TPosition::VECCALC> yFp32Buf_;
    TBuf<TPosition::VECCALC> dyFp32Buf_;
    TBuf<TPosition::VECCALC> zFp32Buf_;
};

} // namespace NsAsinGrad

#endif
