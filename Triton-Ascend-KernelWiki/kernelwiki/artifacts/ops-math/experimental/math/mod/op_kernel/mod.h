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
 * \file mod.h
 * \brief Mod operator implementation
 */

#ifndef MOD_H
#define MOD_H

#include "kernel_operator.h"
#include "mod_tiling_data.h"
#include "mod_tiling_key.h"
#include <limits>

namespace ModNs {
using namespace AscendC;

template <typename T>
class Mod {
public:
    __aicore__ inline Mod(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const ModTilingData* tilingData);
    __aicore__ inline void Process();

    uint8_t bufferNum = 2;
    constexpr static uint8_t DATA_BLOCK = 64;
    constexpr static int QUEUE_DEPTH = 2;

private:
    __aicore__ inline void CopyIn(uint64_t offset, int32_t calCount);
    __aicore__ inline void CopyOut(uint64_t offset, int32_t calCount);
    __aicore__ inline void ParseTilingData(const ModTilingData* tilingData);
    __aicore__ inline void InitConstants();
    __aicore__ inline void ProcessBroadcast(uint64_t inOffset, uint64_t outOffset);
    __aicore__ inline void ProcessContiguous(uint64_t inOffset, uint64_t outOffset);
    __aicore__ inline void Compute(int32_t computeCount);
    __aicore__ inline uint64_t GetInput2Offset(uint64_t outputOffset);
    __aicore__ inline uint32_t GetInput2ContiguousCopyCount(uint64_t outputOffset, uint32_t remainingCount);

    __aicore__ inline void InitBuffers();

    __aicore__ inline void ComputeInt32(
        const int32_t calCount, const int32_t alignedCalCount, LocalTensor<T>& dstTensor, LocalTensor<T>& x1Tensor,
        LocalTensor<T>& x2Tensor, LocalTensor<uint8_t>& sharedTmpBuffer);

    __aicore__ inline void ComputeFPCore(
        const int32_t calCount, const int32_t alignedCalCount, LocalTensor<float>& x1Float, LocalTensor<float>& x2Float,
        LocalTensor<float>& resRem, LocalTensor<float>& resQuot, LocalTensor<uint8_t>& sharedTmpBuffer);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    };

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputx1Queue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputx2Queue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;

    TBuf<TPosition::VECCALC> tmpBuff;

    // Computational Buffers
    TBuf<TPosition::VECCALC> ResQuotTensorBuff;
    TBuf<TPosition::VECCALC> ResRemTensorBuff;

    // Auxiliary Buffers (Constants & Temps)
    TBuf<TPosition::VECCALC> ZeroTensorBuff;
    TBuf<TPosition::VECCALC> InfTensorBuff;
    TBuf<TPosition::VECCALC> NanTensorBuff;
    TBuf<TPosition::VECCALC> MaskTensorBuff;
    TBuf<TPosition::VECCALC> EpsilonTensorBuff;

    // Type Conversion Buffers
    TBuf<TPosition::VECCALC> x1TensorFP32Buff;
    TBuf<TPosition::VECCALC> x2TensorFP32Buff;

    // Int32 High Precision Buffers
    TBuf<TPosition::VECCALC> FP32MaxValidBuff;
    TBuf<TPosition::VECCALC> INT32MaxValidBuff;
    TBuf<TPosition::VECCALC> SplitQuotInt32Buff;
    TBuf<TPosition::VECCALC> SplitRemInt32Buff;

    // Local Tensors (Members to hold handles across functions)
    LocalTensor<float> ResQuotTensor;
    LocalTensor<float> ResRemTensor;
    LocalTensor<float> ZeroTensor;
    LocalTensor<float> InfTensor;
    LocalTensor<float> NanTensor;
    LocalTensor<uint8_t> MaskTensor;
    LocalTensor<float> EpsilonTensor;

    LocalTensor<float> x1TensorFP32Tensor;
    LocalTensor<float> x2TensorFP32Tensor;

    // Int32 Specific Tensors
    LocalTensor<float> FP32MaxValidTensor;
    LocalTensor<int32_t> INT32MaxValidTensor;
    LocalTensor<int32_t> SplitQuotInt32Tensor;
    LocalTensor<int32_t> SplitRemInt32Tensor;

    GlobalTensor<T> inputx1GM;
    GlobalTensor<T> inputx2GM;
    GlobalTensor<T> outputGM;

    // Tiling Parameters
    uint32_t coreNum = 0;
    uint64_t tailCoreNum = 0;
    uint64_t perCoreDataCount = 0;
    uint64_t blockOffset = 0;
    uint32_t blockIdx = 0;
    uint32_t maxDataCount = 0;
    uint32_t actualMaxDataCount = 0;
    uint32_t usableUbSize = 0;
    bool isInput2Scalar = false;
    bool isInput2SameShape = false;
    uint32_t dimNum = 0;
    uint64_t input1Shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t input2Shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t input2Stride[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Constants
    int32_t ShiftParam = 24;
    float infValue = std::numeric_limits<float>::infinity();
    float nanValue = std::numeric_limits<float>::quiet_NaN();
    float Zero = float(0);
    float Epsilon = 1e-20f;
    float FP32MaxValid = 16777216.0f; // 2^24
    int32_t INT32MaxValid = 16777216; // 2^24
};

template <typename T>
__aicore__ inline void Mod<T>::InitBuffers()
{
    pipe.InitBuffer(inputx1Queue, bufferNum, actualMaxDataCount * sizeof(T));
    pipe.InitBuffer(inputx2Queue, bufferNum, actualMaxDataCount * sizeof(T));
    pipe.InitBuffer(outputQueue, bufferNum, actualMaxDataCount * sizeof(T));
    pipe.InitBuffer(tmpBuff, maxDataCount * sizeof(float));

    // 2. 非 Int32 类型 (Float16, Bfloat16, Float32)
    if constexpr (!std::is_same_v<T, int>) {
        pipe.InitBuffer(ResQuotTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(ResRemTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(ZeroTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(InfTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(NanTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(MaskTensorBuff, maxDataCount * sizeof(uint8_t));

        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(x1TensorFP32Buff, maxDataCount * sizeof(float));
            pipe.InitBuffer(x2TensorFP32Buff, maxDataCount * sizeof(float));
        }
    }

    // 3. Int32
    if constexpr (std::is_same_v<T, int>) {
        pipe.InitBuffer(x1TensorFP32Buff, maxDataCount * sizeof(float));
        pipe.InitBuffer(x2TensorFP32Buff, maxDataCount * sizeof(float));

        pipe.InitBuffer(ResQuotTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(ResRemTensorBuff, maxDataCount * sizeof(float));

#if defined(HIGH_PRECISION) && HIGH_PRECISION == 1
        pipe.InitBuffer(FP32MaxValidBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(INT32MaxValidBuff, maxDataCount * sizeof(int32_t));
        pipe.InitBuffer(SplitQuotInt32Buff, maxDataCount * sizeof(int32_t));
        pipe.InitBuffer(SplitRemInt32Buff, maxDataCount * sizeof(int32_t));

        pipe.InitBuffer(EpsilonTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(ZeroTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(MaskTensorBuff, maxDataCount * sizeof(uint8_t));
#endif
    }
}

template <typename T>
__aicore__ inline void Mod<T>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const ModTilingData* tilingData)
{
    inputx1GM.SetGlobalBuffer((__gm__ T*)x1);
    inputx2GM.SetGlobalBuffer((__gm__ T*)x2);
    outputGM.SetGlobalBuffer((__gm__ T*)y);

    ParseTilingData(tilingData);

    // -------------------------------------------------------------
    // Configure Data Tiling Strategy
    // -------------------------------------------------------------
    uint32_t targetMaxData = tilingData->usableUbSize;

    if (perCoreDataCount < targetMaxData) {
        maxDataCount = perCoreDataCount;
        bufferNum = 1;
    } else {
        maxDataCount = targetMaxData;
        bufferNum = 2;
    }

    // Align to DATA_BLOCK
    if (maxDataCount < DATA_BLOCK)
        maxDataCount = DATA_BLOCK;
    maxDataCount = (maxDataCount + DATA_BLOCK - 1) / DATA_BLOCK * DATA_BLOCK;
    actualMaxDataCount = maxDataCount;

    // -------------------------------------------------------------
    // Initialize Buffers
    // -------------------------------------------------------------
    InitBuffers();
}

template <typename T>
__aicore__ inline void Mod<T>::ParseTilingData(const ModTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    coreNum = tilingData->needCoreNum;
    usableUbSize = tilingData->usableUbSize;
    perCoreDataCount = tilingData->perCoreDataCount;
    tailCoreNum = tilingData->tailDataCoreNum;
    isInput2Scalar = tilingData->isInput2Scalar;
    isInput2SameShape = tilingData->isInput2SameShape;
    dimNum = tilingData->dimNum;
    for (uint32_t i = 0; i < 8; ++i) {
        input1Shape[i] = tilingData->input1Shape[i];
        input2Shape[i] = tilingData->input2Shape[i];
        input2Stride[i] = tilingData->input2Stride[i];
    }

    if (tailCoreNum == 0) {
        blockOffset = perCoreDataCount * blockIdx;
    } else {
        if ((blockIdx + 1) <= tailCoreNum) {
            perCoreDataCount += DATA_BLOCK;
            blockOffset = perCoreDataCount * blockIdx;
        } else {
            blockOffset =
                ((perCoreDataCount + DATA_BLOCK) * tailCoreNum) + (perCoreDataCount * (blockIdx - tailCoreNum));
        }
    }

    if (blockIdx == coreNum - 1) {
        perCoreDataCount = tilingData->lastCoreDataCount;
    }
}

template <typename T>
__aicore__ inline void Mod<T>::InitConstants()
{
    if constexpr (std::is_same_v<T, int>) {
#if defined(HIGH_PRECISION) && HIGH_PRECISION == 1
        FP32MaxValidTensor = FP32MaxValidBuff.Get<float>();
        Duplicate(FP32MaxValidTensor, FP32MaxValid, maxDataCount);

        INT32MaxValidTensor = INT32MaxValidBuff.Get<int32_t>();
        Duplicate(INT32MaxValidTensor, INT32MaxValid, maxDataCount);

        ZeroTensor = ZeroTensorBuff.Get<float>();
        Duplicate(ZeroTensor, Zero, maxDataCount);

        EpsilonTensor = EpsilonTensorBuff.Get<float>();
        Duplicate(EpsilonTensor, Epsilon, maxDataCount);
#endif
    } else {
        ZeroTensor = ZeroTensorBuff.Get<float>();
        InfTensor = InfTensorBuff.Get<float>();
        NanTensor = NanTensorBuff.Get<float>();
        MaskTensor = MaskTensorBuff.Get<uint8_t>();

        Duplicate(ZeroTensor, Zero, maxDataCount);
        Duplicate(InfTensor, infValue, maxDataCount);
        Duplicate(NanTensor, nanValue, maxDataCount);
    }
}

template <typename T>
__aicore__ inline void Mod<T>::ProcessBroadcast(uint64_t inOffset, uint64_t outOffset)
{
    uint64_t remainingDataCount = perCoreDataCount;
    while (remainingDataCount > 0) {
        uint32_t currentCount = maxDataCount;
        if (currentCount > remainingDataCount) {
            currentCount = static_cast<uint32_t>(remainingDataCount);
        }
        currentCount = GetInput2ContiguousCopyCount(inOffset, currentCount);
        CopyIn(inOffset, currentCount);
        Compute(currentCount);
        CopyOut(outOffset, currentCount);
        inOffset += currentCount;
        outOffset += currentCount;
        remainingDataCount -= currentCount;
    }
}

template <typename T>
__aicore__ inline void Mod<T>::ProcessContiguous(uint64_t inOffset, uint64_t outOffset)
{
    uint32_t loopCount = perCoreDataCount / maxDataCount;
    uint32_t tailDataCount = perCoreDataCount % maxDataCount;
    for (uint32_t i = 0; i < loopCount; i++) {
        CopyIn(inOffset, maxDataCount);
        Compute(maxDataCount);
        CopyOut(outOffset, maxDataCount);
        inOffset += maxDataCount;
        outOffset += maxDataCount;
    }
    if (tailDataCount > 0) {
        CopyIn(inOffset, tailDataCount);
        Compute(tailDataCount);
        CopyOut(outOffset, tailDataCount);
    }
}

template <typename T>
__aicore__ inline void Mod<T>::Process()
{
    InitConstants();
    if (!isInput2Scalar && !isInput2SameShape) {
        ProcessBroadcast(blockOffset, blockOffset);
    } else {
        ProcessContiguous(blockOffset, blockOffset);
    }
}

template <typename T>
__aicore__ inline uint64_t Mod<T>::GetInput2Offset(const uint64_t outputOffset)
{
    uint64_t remaining = outputOffset;
    uint64_t input2Offset = 0;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        uint64_t coord = 0;
        uint64_t dimSize = input1Shape[i];
        if (dimSize > 0) {
            coord = remaining % dimSize;
            remaining = remaining / dimSize;
        }
        input2Offset += coord * input2Stride[i];
    }
    return input2Offset;
}

template <typename T>
__aicore__ inline uint32_t Mod<T>::GetInput2ContiguousCopyCount(
    const uint64_t outputOffset, const uint32_t remainingCount)
{
    uint64_t suffixSize = 1;
    uint64_t expectedStride = 1;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        if (input1Shape[i] == 1) {
            continue;
        }
        if (input2Stride[i] != expectedStride) {
            break;
        }
        suffixSize *= input1Shape[i];
        expectedStride *= input2Shape[i];
    }
    uint64_t count = suffixSize - (outputOffset % suffixSize);
    if (count > remainingCount) {
        count = remainingCount;
    }
    return static_cast<uint32_t>(count);
}

template <typename T>
__aicore__ inline void Mod<T>::CopyIn(const uint64_t offset, const int32_t calCount)
{
    LocalTensor<T> datax1Local = inputx1Queue.AllocTensor<T>();
    LocalTensor<T> datax2Local = inputx2Queue.AllocTensor<T>();
    const int32_t alignedCalCount = CeilAlign(calCount, DATA_BLOCK);

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = calCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(datax1Local, inputx1GM[offset], copyParams, {false, 0, 0, 0});
    if (isInput2Scalar) {
        DataCopyParams scalarCopyParams;
        scalarCopyParams.blockCount = 1;
        scalarCopyParams.blockLen = sizeof(T);
        scalarCopyParams.srcStride = 0;
        scalarCopyParams.dstStride = 0;
        DataCopyPad(datax2Local, inputx2GM[0], scalarCopyParams, {false, 0, 0, 0});
        T scalarValue = datax2Local.GetValue(0);
        Duplicate(datax2Local, scalarValue, alignedCalCount);
    } else if (!isInput2SameShape) {
        DataCopyParams input2CopyParams;
        input2CopyParams.blockCount = 1;
        input2CopyParams.blockLen = calCount * sizeof(T);
        input2CopyParams.srcStride = 0;
        input2CopyParams.dstStride = 0;
        DataCopyPad(datax2Local, inputx2GM[GetInput2Offset(offset)], input2CopyParams, {false, 0, 0, 0});
    } else {
        DataCopyPad(datax2Local, inputx2GM[offset], copyParams, {false, 0, 0, 0});
    }

    inputx1Queue.EnQue(datax1Local);
    inputx2Queue.EnQue(datax2Local);
}

template <typename T>
__aicore__ inline void Mod<T>::CopyOut(const uint64_t offset, const int32_t calCount)
{
    LocalTensor<T> dstLocal = outputQueue.DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = calCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGM[offset], dstLocal, copyParams);

    outputQueue.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void Mod<T>::ComputeInt32(
    const int32_t calCount, const int32_t alignedCalCount, LocalTensor<T>& dstTensor, LocalTensor<T>& x1Tensor,
    LocalTensor<T>& x2Tensor, LocalTensor<uint8_t>& sharedTmpBuffer)
{
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    x1TensorFP32Tensor = x1TensorFP32Buff.Get<float>();
    x2TensorFP32Tensor = x2TensorFP32Buff.Get<float>();
    ResQuotTensor = ResQuotTensorBuff.Get<float>();
    ResRemTensor = ResRemTensorBuff.Get<float>();

    Cast(x1TensorFP32Tensor, x1Tensor, AscendC::RoundMode::CAST_NONE, calCount);
    Cast(x2TensorFP32Tensor, x2Tensor, AscendC::RoundMode::CAST_NONE, calCount);

    Div(ResRemTensor, x1TensorFP32Tensor, x2TensorFP32Tensor, calCount);
    Floor(ResQuotTensor, ResRemTensor, sharedTmpBuffer, calCount);
    Mul(ResQuotTensor, ResQuotTensor, x2TensorFP32Tensor, calCount);
    Sub(ResRemTensor, x1TensorFP32Tensor, ResQuotTensor, calCount);

    Cast(dstTensor, ResRemTensor, AscendC::RoundMode::CAST_RINT, calCount);

#else

    FP32MaxValidTensor = FP32MaxValidBuff.Get<float>();
    INT32MaxValidTensor = INT32MaxValidBuff.Get<int32_t>();
    EpsilonTensor = EpsilonTensorBuff.Get<float>();

    x2TensorFP32Tensor = x2TensorFP32Buff.Get<float>();
    SplitRemInt32Tensor = SplitRemInt32Buff.Get<int32_t>();
    SplitQuotInt32Tensor = SplitQuotInt32Buff.Get<int32_t>();

    LocalTensor<float> q1FloatTensor = ResQuotTensorBuff.Get<float>();
    LocalTensor<float> q2FloatTensor = ResRemTensorBuff.Get<float>();
    LocalTensor<int32_t> q2IntTensor = dstTensor;

    Cast(x2TensorFP32Tensor, x2Tensor, AscendC::RoundMode::CAST_NONE, calCount);
    Add(x2TensorFP32Tensor, x2TensorFP32Tensor, EpsilonTensor, calCount);
    Div(q2FloatTensor, FP32MaxValidTensor, x2TensorFP32Tensor, calCount);
    ShiftRight(SplitQuotInt32Tensor, x1Tensor, 24, calCount);
    ShiftLeft(SplitRemInt32Tensor, SplitQuotInt32Tensor, 24, calCount);
    Sub(SplitRemInt32Tensor, x1Tensor, SplitRemInt32Tensor, calCount);
    Floor(q2FloatTensor, q2FloatTensor, sharedTmpBuffer, calCount);
    Cast(q2IntTensor, q2FloatTensor, AscendC::RoundMode::CAST_RINT, calCount);
    Mul(q2IntTensor, q2IntTensor, x2Tensor, calCount);
    Sub(q2IntTensor, INT32MaxValidTensor, q2IntTensor, calCount);
    Mul(SplitQuotInt32Tensor, SplitQuotInt32Tensor, q2IntTensor, calCount);
    Add(SplitQuotInt32Tensor, SplitQuotInt32Tensor, SplitRemInt32Tensor, calCount);
    Cast(q1FloatTensor, SplitQuotInt32Tensor, AscendC::RoundMode::CAST_NONE, calCount);
    Div(q1FloatTensor, q1FloatTensor, x2TensorFP32Tensor, calCount);
    Floor(q1FloatTensor, q1FloatTensor, sharedTmpBuffer, calCount);
    Cast(SplitRemInt32Tensor, q1FloatTensor, AscendC::RoundMode::CAST_RINT, calCount);
    Mul(SplitRemInt32Tensor, SplitRemInt32Tensor, x2Tensor, calCount);
    Sub(dstTensor, SplitQuotInt32Tensor, SplitRemInt32Tensor, calCount);
#endif
}

template <typename T>
__aicore__ inline void Mod<T>::ComputeFPCore(
    const int32_t calCount, const int32_t alignedCalCount, LocalTensor<float>& x1Float, LocalTensor<float>& x2Float,
    LocalTensor<float>& resRem, LocalTensor<float>& resQuot, LocalTensor<uint8_t>& sharedTmpBuffer)
{
    Div(resRem, x1Float, x2Float, calCount);
    Trunc(resQuot, resRem, sharedTmpBuffer, calCount);
    Mul(resQuot, resQuot, x2Float, calCount);
    Sub(resRem, x1Float, resQuot, calCount);

    Abs(resQuot, x2Float, calCount);
    Compare(MaskTensor, resQuot, InfTensor, AscendC::CMPMODE::EQ, alignedCalCount);
    Select(resRem, MaskTensor, x1Float, resRem, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCalCount);

    Abs(resQuot, x1Float, calCount);
    Compare(MaskTensor, resQuot, InfTensor, AscendC::CMPMODE::EQ, alignedCalCount);
    Select(resRem, MaskTensor, NanTensor, resRem, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCalCount);
}

template <typename T>
__aicore__ inline void Mod<T>::Compute(const int32_t calCount)
{
    LocalTensor<T> x1Tensor = inputx1Queue.DeQue<T>();
    LocalTensor<T> x2Tensor = inputx2Queue.DeQue<T>();
    LocalTensor<T> dstTensor = outputQueue.AllocTensor<T>();
    LocalTensor<uint8_t> sharedTmpBuffer = tmpBuff.Get<uint8_t>();

    int32_t alignedCalCount = (calCount + DATA_BLOCK - 1) / DATA_BLOCK * DATA_BLOCK;

    // Int32
    if constexpr (std::is_same_v<T, int>) {
        ComputeInt32(calCount, alignedCalCount, dstTensor, x1Tensor, x2Tensor, sharedTmpBuffer);
    }
    // Half, Bfloat16, Float32
    else {
        ResQuotTensor = ResQuotTensorBuff.Get<float>();
        ResRemTensor = ResRemTensorBuff.Get<float>();

        // FP16 (Half) & Bfloat16
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            x1TensorFP32Tensor = x1TensorFP32Buff.Get<float>();
            x2TensorFP32Tensor = x2TensorFP32Buff.Get<float>();

            Cast(x1TensorFP32Tensor, x1Tensor, AscendC::RoundMode::CAST_NONE, alignedCalCount);
            Cast(x2TensorFP32Tensor, x2Tensor, AscendC::RoundMode::CAST_NONE, alignedCalCount);

            ComputeFPCore(
                calCount, alignedCalCount, x1TensorFP32Tensor, x2TensorFP32Tensor, ResRemTensor, ResQuotTensor,
                sharedTmpBuffer);

            if constexpr (std::is_same_v<T, half>) {
                Cast(dstTensor, ResRemTensor, AscendC::RoundMode::CAST_NONE, calCount);
            } else {
                Cast(dstTensor, ResRemTensor, AscendC::RoundMode::CAST_RINT, calCount);
            }
        }
        // Float32
        else {
            ComputeFPCore(calCount, alignedCalCount, x1Tensor, x2Tensor, ResRemTensor, ResQuotTensor, sharedTmpBuffer);
            Add(dstTensor, ResRemTensor, ZeroTensor, calCount);
        }
    }

    inputx1Queue.FreeTensor(x1Tensor);
    inputx2Queue.FreeTensor(x2Tensor);
    outputQueue.EnQue(dstTensor);
}

template <int D_T_X1, int D_T_X2, int D_T_Y>
__aicore__ inline void ModKernelImpl(
    __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y, const ModNs::ModTilingData* tilingData)
{
    GM_ADDR userWS = nullptr;

    if constexpr (D_T_X1 == MOD_TPL_INT32 && D_T_X2 == MOD_TPL_INT32 && D_T_Y == MOD_TPL_INT32) {
        ModNs::Mod<int> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
    } else if constexpr (D_T_X1 == MOD_TPL_FP16 && D_T_X2 == MOD_TPL_FP16 && D_T_Y == MOD_TPL_FP16) {
        ModNs::Mod<half> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
    } else if constexpr (D_T_X1 == MOD_TPL_FP32 && D_T_X2 == MOD_TPL_FP32 && D_T_Y == MOD_TPL_FP32) {
        ModNs::Mod<float> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if constexpr (D_T_X1 == MOD_TPL_BF16 && D_T_X2 == MOD_TPL_BF16 && D_T_Y == MOD_TPL_BF16) {
        ModNs::Mod<bfloat16_t> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
#endif
    }
}

} // namespace ModNs
#endif // MOD_H
