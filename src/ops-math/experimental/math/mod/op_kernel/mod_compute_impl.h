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
 * \file mod_compute_impl.h
 * \brief Mod<T, ST, OT> per-tile same-dtype compute definitions.
 *
 * This file is included from mod.h inside namespace ModNs.
 */
#ifndef MOD_COMPUTE_IMPL_H
#define MOD_COMPUTE_IMPL_H

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ComputeFPCore(const int32_t calCount, const int32_t alignedCalCount,
                                                     LocalTensor<float>& x1Float, LocalTensor<float>& x2Float,
                                                     LocalTensor<float>& resRem, LocalTensor<float>& resQuot,
                                                     LocalTensor<uint8_t>& sharedTmpBuffer)
{
#if MOD_ENH_ARCH22
    // K1: fp32/fp16/bf16 use adaptive naive/AlgoA routing. K2: int16 always uses the exact-width naive route.
    (void)sharedTmpBuffer;
    LocalTensor<float> w0 = resQuot;
    if constexpr (USE_ALGO_A) {
        LocalTensor<float> w1 = A1Buff.Get<float>();
        LocalTensor<float> w2 = A2Buff.Get<float>();
        LocalTensor<float> w3 = A3Buff.Get<float>();
        LocalTensor<float> w4 = A4Buff.Get<float>();
        LocalTensor<float> w5 = A5Buff.Get<float>();
        RemainderRouted(resRem, x1Float, x2Float, w0, w1, w2, w3, w4, w5, calCount);
    } else {
        Div(w0, x1Float, x2Float, calCount);
        Cast(w0, w0, AscendC::RoundMode::CAST_TRUNC, calCount);
        Mul(w0, w0, x2Float, calCount);
        Sub(resRem, x1Float, w0, calCount);
    }
#else
    Div(resRem, x1Float, x2Float, calCount);
    Trunc(resQuot, resRem, sharedTmpBuffer, calCount);
    Mul(resQuot, resQuot, x2Float, calCount);
    Sub(resRem, x1Float, resQuot, calCount);
#endif

    Abs(resQuot, x2Float, calCount);
    Compare(MaskTensor, resQuot, InfTensor, AscendC::CMPMODE::EQ, alignedCalCount);
    Select(resRem, MaskTensor, x1Float, resRem, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCalCount);

    Abs(resQuot, x1Float, calCount);
    Compare(MaskTensor, resQuot, InfTensor, AscendC::CMPMODE::EQ, alignedCalCount);
    Select(resRem, MaskTensor, NanTensor, resRem, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCalCount);
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ComputeCore(const int32_t calCount, LocalTensor<ST>& x1Tensor,
                                                   LocalTensor<OT>& x2Tensor, LocalTensor<T>& dstTensor,
                                                   LocalTensor<uint8_t>& sharedTmpBuffer)
{
    int32_t alignedCalCount = (calCount + DATA_BLOCK - 1) / DATA_BLOCK * DATA_BLOCK;

    if constexpr (std::is_same_v<T, int>) {
        ComputeInt32(calCount, alignedCalCount, dstTensor, x1Tensor, x2Tensor, sharedTmpBuffer);
    } else {
        ResQuotTensor = ResQuotTensorBuff.Get<float>();
        ResRemTensor = ResRemTensorBuff.Get<float>();

        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int16_t>) {
            x1TensorFP32Tensor = x1TensorFP32Buff.Get<float>();
            x2TensorFP32Tensor = x2TensorFP32Buff.Get<float>();

            Cast(x1TensorFP32Tensor, x1Tensor, AscendC::RoundMode::CAST_NONE, alignedCalCount);
            Cast(x2TensorFP32Tensor, x2Tensor, AscendC::RoundMode::CAST_NONE, alignedCalCount);

            ComputeFPCore(calCount, alignedCalCount, x1TensorFP32Tensor, x2TensorFP32Tensor, ResRemTensor,
                          ResQuotTensor, sharedTmpBuffer);

            if constexpr (std::is_same_v<T, half>) {
                Cast(dstTensor, ResRemTensor, AscendC::RoundMode::CAST_NONE, calCount);
            } else {
                Cast(dstTensor, ResRemTensor, AscendC::RoundMode::CAST_RINT, calCount);
            }
        } else {
            ComputeFPCore(calCount, alignedCalCount, x1Tensor, x2Tensor, ResRemTensor, ResQuotTensor, sharedTmpBuffer);
            Add(dstTensor, ResRemTensor, ZeroTensor, calCount);
        }
    }
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::Compute(const int32_t calCount)
{
    LocalTensor<ST> x1Tensor = inputx1Queue.DeQue<ST>();
    LocalTensor<OT> x2Tensor = inputx2Queue.DeQue<OT>();
    LocalTensor<T> dstTensor = outputQueue.AllocTensor<T>();
    LocalTensor<uint8_t> sharedTmpBuffer = tmpBuff.Get<uint8_t>();

    ComputeCore(calCount, x1Tensor, x2Tensor, dstTensor, sharedTmpBuffer);

    inputx1Queue.FreeTensor(x1Tensor);
    inputx2Queue.FreeTensor(x2Tensor);
    outputQueue.EnQue(dstTensor);
}

#endif // MOD_COMPUTE_IMPL_H
