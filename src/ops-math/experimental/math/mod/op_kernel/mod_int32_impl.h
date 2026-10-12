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
 * \file mod_int32_impl.h
 * \brief Mod<T,ST,OT>::ComputeInt32 (upstream int32 2^24 high-precision split-multiply path).
 *
 * ComputeInt32 为上游所有 (int32 2^24 高精度 split-multiply 路径，K1/K2 不触及)、自包含，拆出到本文件
 * (纯物理迁移，无逻辑改动)。同 mod_flat_impl.h：不自带 namespace，从 mod.h 的 `namespace ModNs` 内 #include
 * -> 定义附着到 ModNs::Mod<T,ST,OT>。
 */
#ifndef MOD_INT32_IMPL_H
#define MOD_INT32_IMPL_H

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ComputeInt32(const int32_t calCount, const int32_t alignedCalCount,
                                                    LocalTensor<T>& dstTensor, LocalTensor<ST>& x1Tensor,
                                                    LocalTensor<OT>& x2Tensor, LocalTensor<uint8_t>& sharedTmpBuffer)
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

#endif // MOD_INT32_IMPL_H
