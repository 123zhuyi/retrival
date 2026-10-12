/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IS_NEG_INF_H
#define IS_NEG_INF_H

#include "kernel_operator.h"

namespace NsIsNegInf {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t CMP_ALIGN_BYTES = 256;

template <typename T>
struct KernelContext {
    TPipe* pipe = nullptr;
    GlobalTensor<T> xGm;
    GlobalTensor<uint8_t> yGm;
    int64_t blockLength = 0;
    int64_t tileLength = 0;
};

template <typename T>
__aicore__ inline void InitKernelContext(KernelContext<T>& context,
                                         GM_ADDR x,
                                         GM_ADDR y,
                                         int64_t formerNum,
                                         int64_t formerLength,
                                         int64_t tailLength,
                                         int64_t tileLength,
                                         TPipe* pipeIn)
{
    context.pipe = pipeIn;
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx < formerNum) {
        context.blockLength = formerLength;
        int64_t offset = formerLength * blockIdx;
        context.xGm.SetGlobalBuffer((__gm__ T*)x + offset, formerLength);
        context.yGm.SetGlobalBuffer((__gm__ uint8_t*)y + offset, formerLength);
    } else {
        context.blockLength = tailLength;
        int64_t offset = formerLength * formerNum;
        context.xGm.SetGlobalBuffer((__gm__ T*)x + offset, tailLength);
        context.yGm.SetGlobalBuffer((__gm__ uint8_t*)y + offset, tailLength);
    }
    context.tileLength = tileLength;
}

template <typename T>
__aicore__ inline void CopyInTile(TQue<TPosition::VECIN, BUFFER_NUM>& inQueueX,
                                  GlobalTensor<T>& xGm,
                                  int64_t progress,
                                  int64_t tileLength,
                                  int64_t validLength)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, xGm[progress * tileLength], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void FinishCompute(LocalTensor<T>& xLocal,
                                     LocalTensor<uint8_t>& outLocal,
                                     TQue<TPosition::VECIN, BUFFER_NUM>& inQueueX,
                                     TQue<TPosition::VECOUT, BUFFER_NUM>& outQueueY)
{
    outQueueY.EnQue<uint8_t>(outLocal);
    inQueueX.FreeTensor(xLocal);
}

__aicore__ inline void CopyOutTile(TQue<TPosition::VECOUT, BUFFER_NUM>& outQueueY,
                                   GlobalTensor<uint8_t>& yGm,
                                   int64_t progress,
                                   int64_t tileLength,
                                   int64_t validLength)
{
    LocalTensor<uint8_t> outLocal = outQueueY.DeQue<uint8_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad(yGm[progress * tileLength], outLocal, copyParams);
    outQueueY.FreeTensor(outLocal);
}

__aicore__ inline void BuildMaskOutput(LocalTensor<uint8_t>& maskLocal,
                                       LocalTensor<half>& halfLocal,
                                       LocalTensor<half>& oneLocal,
                                       LocalTensor<uint8_t>& outLocal,
                                       uint32_t computeLength)
{
    Duplicate(oneLocal, static_cast<half>(1), computeLength);
    PipeBarrier<PIPE_V>();
    Select<half, uint8_t>(halfLocal, maskLocal, oneLocal, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLength);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, halfLocal, RoundMode::CAST_NONE, computeLength);
}

template <typename KernelT>
__aicore__ inline void ProcessTiles(KernelT* kernel, int64_t blockLength, int64_t tileLength)
{
    int64_t tileNum = (blockLength + tileLength - 1) / tileLength;
    if (tileNum == 0) {
        return;
    }
    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t validLength = (i == tileNum - 1) ? (blockLength - (tileNum - 1) * tileLength) : tileLength;
        kernel->CopyIn(i, validLength);
        kernel->Compute(validLength);
        kernel->CopyOut(i, validLength);
    }
}

template <typename T>
__aicore__ inline T NegInfValue();

template <>
__aicore__ inline float NegInfValue<float>()
{
    return -1.0f / 0.0f;
}

template <>
__aicore__ inline half NegInfValue<half>()
{
    return static_cast<half>(-1.0f / 0.0f);
}

template <>
__aicore__ inline bfloat16_t NegInfValue<bfloat16_t>()
{
    return static_cast<bfloat16_t>(-1.0f / 0.0f);
}

template <typename T, typename Derived>
class KernelIsNegInfBase
{
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, int64_t formerNum, int64_t formerLength, int64_t tailLength, int64_t tileLength, TPipe* pipeIn)
    {
        InitKernelContext(context, x, y, formerNum, formerLength, tailLength, tileLength, pipeIn);
        context.pipe->InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        context.pipe->InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(uint8_t));
        static_cast<Derived*>(this)->InitComputeBuffers(tileLength);
    }

    __aicore__ inline void Process()
    {
        ProcessTiles(this, context.blockLength, context.tileLength);
    }

    __aicore__ inline void CopyIn(int64_t progress, int64_t validLength)
    {
        CopyInTile(inQueueX, context.xGm, progress, context.tileLength, validLength);
    }

    __aicore__ inline void Compute(int64_t validLength)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueueY.AllocTensor<uint8_t>();
        static_cast<Derived*>(this)->ComputeImpl(xLocal, outLocal, validLength);
        FinishCompute(xLocal, outLocal, inQueueX, outQueueY);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t validLength)
    {
        CopyOutTile(outQueueY, context.yGm, progress, context.tileLength, validLength);
    }

protected:
    KernelContext<T> context;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
};

template <typename T>
class KernelIsNegInf : public KernelIsNegInfBase<T, KernelIsNegInf<T>>
{
public:
    __aicore__ inline void InitComputeBuffers(int64_t tileLength)
    {
        this->context.pipe->InitBuffer(tmpBufCmp, tileLength * sizeof(T));
    }

    __aicore__ inline void ComputeImpl(LocalTensor<T>& xLocal, LocalTensor<uint8_t>& outLocal, int64_t validLength)
    {
        LocalTensor<T> cmpLocal = tmpBufCmp.Get<T>();
        if constexpr (std::is_same_v<T, half>) {
            uint32_t alignCount = CMP_ALIGN_BYTES / sizeof(T);
            uint32_t computeLength = static_cast<uint32_t>(((validLength + alignCount - 1) / alignCount) * alignCount);
            Duplicate(cmpLocal, NegInfValue<T>(), computeLength);
            PipeBarrier<PIPE_V>();
            Compare(outLocal, xLocal, cmpLocal, CMPMODE::EQ, computeLength);
            PipeBarrier<PIPE_V>();
            Duplicate<T>(xLocal, static_cast<T>(1), computeLength);
            PipeBarrier<PIPE_V>();
            Select(xLocal, outLocal, xLocal, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLength);
            PipeBarrier<PIPE_V>();
            Cast(outLocal, xLocal, RoundMode::CAST_NONE, computeLength);
        } else {
            T negInf = NegInfValue<T>();
            for (int64_t idx = 0; idx < validLength; ++idx) {
                T value = xLocal.GetValue(idx);
                outLocal.SetValue(idx, value == negInf ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
            }
        }
    }

private:
    TBuf<TPosition::VECCALC> tmpBufCmp;
};

class KernelIsNegInfFp32Unified : public KernelIsNegInfBase<float, KernelIsNegInfFp32Unified>
{
public:
    __aicore__ inline void InitComputeBuffers(int64_t tileLength)
    {
        this->context.pipe->InitBuffer(tmpBufCmp, tileLength * sizeof(float));
        this->context.pipe->InitBuffer(tmpBufMask, tileLength * sizeof(uint8_t));
        this->context.pipe->InitBuffer(tmpBufHalf, tileLength * sizeof(half));
        this->context.pipe->InitBuffer(tmpBufOne, tileLength * sizeof(half));
    }

    __aicore__ inline void ComputeImpl(LocalTensor<float>& xLocal, LocalTensor<uint8_t>& outLocal, int64_t validLength)
    {
        LocalTensor<float> cmpLocal = tmpBufCmp.Get<float>();
        LocalTensor<uint8_t> maskLocal = tmpBufMask.Get<uint8_t>();
        LocalTensor<half> halfLocal = tmpBufHalf.Get<half>();
        LocalTensor<half> oneLocal = tmpBufOne.Get<half>();

        uint32_t alignCount = CMP_ALIGN_BYTES / sizeof(float);
        uint32_t computeLength = static_cast<uint32_t>(((validLength + alignCount - 1) / alignCount) * alignCount);
        Duplicate(cmpLocal, NegInfValue<float>(), computeLength);
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, xLocal, cmpLocal, CMPMODE::EQ, computeLength);
        BuildMaskOutput(maskLocal, halfLocal, oneLocal, outLocal, computeLength);
    }

private:
    TBuf<TPosition::VECCALC> tmpBufCmp;
    TBuf<TPosition::VECCALC> tmpBufMask;
    TBuf<TPosition::VECCALC> tmpBufHalf;
    TBuf<TPosition::VECCALC> tmpBufOne;
};

class KernelIsNegInfBf16Unified : public KernelIsNegInfBase<bfloat16_t, KernelIsNegInfBf16Unified>
{
public:
    __aicore__ inline void InitComputeBuffers(int64_t tileLength)
    {
        this->context.pipe->InitBuffer(tmpBufFp32, tileLength * sizeof(float));
        this->context.pipe->InitBuffer(tmpBufCmp, tileLength * sizeof(float));
        this->context.pipe->InitBuffer(tmpBufMask, tileLength * sizeof(uint8_t));
        this->context.pipe->InitBuffer(tmpBufHalf, tileLength * sizeof(half));
        this->context.pipe->InitBuffer(tmpBufOne, tileLength * sizeof(half));
    }

    __aicore__ inline void ComputeImpl(LocalTensor<bfloat16_t>& xLocal, LocalTensor<uint8_t>& outLocal, int64_t validLength)
    {
        LocalTensor<float> fp32Local = tmpBufFp32.Get<float>();
        LocalTensor<float> cmpLocal = tmpBufCmp.Get<float>();
        LocalTensor<uint8_t> maskLocal = tmpBufMask.Get<uint8_t>();
        LocalTensor<half> halfLocal = tmpBufHalf.Get<half>();
        LocalTensor<half> oneLocal = tmpBufOne.Get<half>();

        uint32_t alignCount = CMP_ALIGN_BYTES / sizeof(float);
        uint32_t computeLength = static_cast<uint32_t>(((validLength + alignCount - 1) / alignCount) * alignCount);
        Cast(fp32Local, xLocal, RoundMode::CAST_NONE, computeLength);
        Duplicate(cmpLocal, NegInfValue<float>(), computeLength);
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, fp32Local, cmpLocal, CMPMODE::EQ, computeLength);
        BuildMaskOutput(maskLocal, halfLocal, oneLocal, outLocal, computeLength);
    }

private:
    TBuf<TPosition::VECCALC> tmpBufFp32;
    TBuf<TPosition::VECCALC> tmpBufCmp;
    TBuf<TPosition::VECCALC> tmpBufMask;
    TBuf<TPosition::VECCALC> tmpBufHalf;
    TBuf<TPosition::VECCALC> tmpBufOne;
};

} // namespace NsIsNegInf

#endif
