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
 * \file vf_rope.h
 * \brief
 */
 
#ifndef VF_ROPE_H
#define VF_ROPE_H
#include "kernel_tensor.h"

#if __CCE_AICORE__ == 310

namespace MlaProlog {
constexpr uint64_t FLOAT_VF_SIZE = 64;

template <typename T>
__simd_vf__ void RopeVFImpl(__ubuf__ T * ropeUb, __ubuf__ T * sinUb, __ubuf__ T * cosUb, __ubuf__ uint32_t * gatherUb1, 
    __ubuf__ uint32_t * gatherUb2, __ubuf__ T * resUb, const uint16_t row)
{
    MicroAPI::RegTensor<float> vregRopeFp32_1;
    MicroAPI::RegTensor<float> vregRopeFp32_2;
    MicroAPI::RegTensor<uint32_t> vregIndex_1;
    MicroAPI::RegTensor<uint32_t> vregIndex_2;
    MicroAPI::RegTensor<T> vregRes_1;
    MicroAPI::RegTensor<T> vregSinMulLow;
    MicroAPI::RegTensor<T> vregSinMulHigh;
    MicroAPI::RegTensor<T> vregRes_2;
    MicroAPI::RegTensor<T> vregSin;
    MicroAPI::RegTensor<T> vregSinDouble;
    MicroAPI::RegTensor<T> vregCos;
    MicroAPI::RegTensor<T> vregCosDouble;
    MicroAPI::MaskReg preg_all = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t halfReg = 32;
    MicroAPI::MaskReg maskLower64 = MicroAPI::UpdateMask<T>(halfReg);
    MicroAPI::MaskReg maskHigher64;
    MicroAPI::Xor(maskHigher64, maskLower64, preg_all, preg_all);
    // 奇数在前，偶数在后
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregIndex_1, gatherUb1);
    // 偶数在前，奇数在后
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregIndex_2, gatherUb2);

    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregCosDouble, cosUb);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSinDouble, sinUb);

    static constexpr MicroAPI::CastTrait castTrait ={MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait castTrait0 ={MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    
    for (uint16_t i = 0; i < row; i++) {
        MicroAPI::Gather(vregRopeFp32_1, ropeUb + i * FLOAT_VF_SIZE, vregIndex_1, preg_all);
        MicroAPI::Gather(vregRopeFp32_2, ropeUb + i * FLOAT_VF_SIZE, vregIndex_2, preg_all);
        MicroAPI::Mul(vregRes_2, vregCosDouble, vregRopeFp32_2, preg_all);

        MicroAPI::Muls(vregSinMulLow, vregRopeFp32_1, 1, maskLower64);
        MicroAPI::Muls(vregSinMulHigh, vregRopeFp32_1, 1, maskHigher64);
        MicroAPI::Add(vregRes_1, vregSinMulHigh, vregSinMulLow, preg_all);

        MicroAPI::Mul(vregRes_1, vregSinDouble, vregRes_1, preg_all);

        MicroAPI::Add(vregRes_2, vregRes_1, vregRes_2, preg_all);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(resUb + i * FLOAT_VF_SIZE, vregRes_2, preg_all);
    }
}

template <typename T>
__aicore__ inline void RopeVF(const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor, const LocalTensor<T>& xTensor,
    const LocalTensor<uint32_t>& gatherTensor1, const LocalTensor<uint32_t>& gatherTensor2,
    const LocalTensor<T>& resTensor, const uint16_t row)
{
    __ubuf__ T * ropeUb = (__ubuf__ T*)xTensor.GetPhyAddr();
    __ubuf__ T * sinUb = (__ubuf__ T*)sinTensor.GetPhyAddr();
    __ubuf__ T * cosUb = (__ubuf__ T*)cosTensor.GetPhyAddr();
    __ubuf__ uint32_t * gatherUb1 = (__ubuf__ uint32_t*)gatherTensor1.GetPhyAddr();
    __ubuf__ uint32_t * gatherUb2 = (__ubuf__ uint32_t*)gatherTensor2.GetPhyAddr();
    __ubuf__ T * resUb = (__ubuf__ T*)resTensor.GetPhyAddr();

    RopeVFImpl<T>(ropeUb, sinUb, cosUb, gatherUb1, gatherUb2, resUb, row);
}

template <typename C>
__aicore__ inline void RotaryPosEmbVF(const LocalTensor<C> &outputLocal, const LocalTensor<C> &inputLocal, const LocalTensor<C> &cosLocal,
                                    const LocalTensor<C> &sinLocal, const LocalTensor<uint8_t> &shareTmpUb, uint64_t row, uint64_t col) {
    uint64_t cnt = row * col;
    uint32_t offsetByBytes = 0;
    constexpr uint32_t halfReg = 32;

    LocalTensor<uint32_t> gatherTensor1 = shareTmpUb.ReinterpretCast<uint32_t>()[offsetByBytes / sizeof(uint32_t)];
    offsetByBytes += col * sizeof(uint32_t); // 偏移col
    LocalTensor<uint32_t> gatherTensor2 = shareTmpUb.ReinterpretCast<uint32_t>()[offsetByBytes / sizeof(uint32_t)];
    offsetByBytes += col * sizeof(uint32_t); // 偏移col

    for(int i = 0; i < halfReg; ++i) {
        gatherTensor1.SetValue(i, i * 2 + 1); // 奇数在前面32
        gatherTensor1.SetValue(i + halfReg, i * 2); // 偶数在后面32
        gatherTensor2.SetValue(i, i * 2); // 偶数在前面32
        gatherTensor2.SetValue(i + halfReg, i * 2 + 1); // 奇数在后面32
    }
    RopeVF<C>(sinLocal, cosLocal, inputLocal, gatherTensor1, gatherTensor2, outputLocal, static_cast<uint16_t>(row));
}

} // namespace MlaProlog
#endif // __CCE_AICORE__ == 310
#endif // VF_ROPE_H