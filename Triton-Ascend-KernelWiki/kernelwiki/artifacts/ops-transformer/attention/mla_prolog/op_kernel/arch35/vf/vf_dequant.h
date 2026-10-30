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
 * \file vf_dequant.h
 * \brief 
 */

#ifndef VF_DEQUANT_H
#define VF_DEQUANT_H
#include "kernel_tensor.h"

namespace MlaProlog {

template<typename T, typename C, typename O>
__simd_vf__ void DequantVFImpl (__ubuf__ O* yAddr, __ubuf__ T* xAddr, __ubuf__ C * scalePerChannelAddr, __ubuf__ C * scalePerTokenAddr,
    uint32_t floatRepSize, uint32_t fp32BlockElementNum, uint32_t dLoops, uint32_t dTail, uint32_t dTailLoop, uint32_t row, uint32_t col, uint32_t stride)
{
    constexpr static AscendC::MicroAPI::CastTrait castTraitInt32ToFp32 = {
        AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    AscendC::MicroAPI::RegTensor<T> vregInput;
    AscendC::MicroAPI::RegTensor<C> vregScalePerChannel;
    AscendC::MicroAPI::RegTensor<C> vregScalePerToken;
    AscendC::MicroAPI::RegTensor<float> vregInputFp32; // cast成float之后的vregInput
    AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg tailMask;
    tailMask = AscendC::MicroAPI::UpdateMask<float>(dTail);

    uint32_t colOffset = 0;
    uint32_t rowOffset = 0;
    uint32_t scaleOffset = 0;
    for (uint32_t j = 0; j < dLoops; j++) {
        AscendC::MicroAPI::LoadAlign<C, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregScalePerChannel, scalePerChannelAddr + colOffset);
        rowOffset = 0;
        scaleOffset = 0;
        for (uint32_t i = 0; i < row; i++) {
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregInput, xAddr + colOffset + rowOffset);
            AscendC::MicroAPI::LoadAlign<C, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vregScalePerToken, scalePerTokenAddr + scaleOffset);
            AscendC::MicroAPI::Cast<float, T, castTraitInt32ToFp32>(vregInputFp32, vregInput, fullMask);
            AscendC::MicroAPI::Mul(vregInputFp32, vregInputFp32, vregScalePerChannel, fullMask);
            AscendC::MicroAPI::Mul(vregInputFp32, vregInputFp32, vregScalePerToken, fullMask);
            AscendC::MicroAPI::StoreAlign<O, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(yAddr + colOffset + rowOffset, vregInputFp32, fullMask);
            rowOffset += stride;
            scaleOffset += fp32BlockElementNum;
        }
        colOffset += floatRepSize;
    }
   
    if (dTailLoop > 0) {
        rowOffset = 0;
        scaleOffset = 0;
        AscendC::MicroAPI::LoadAlign<C, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregScalePerChannel, scalePerChannelAddr + dLoops * floatRepSize);
        for (uint32_t i = 0; i < row; i++) {
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregInput, xAddr + dLoops * floatRepSize + rowOffset);
            AscendC::MicroAPI::LoadAlign<C, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vregScalePerToken, scalePerTokenAddr + scaleOffset);
            AscendC::MicroAPI::Cast<float, T, castTraitInt32ToFp32>(vregInputFp32, vregInput, tailMask);
            AscendC::MicroAPI::Mul(vregInputFp32, vregInputFp32, vregScalePerChannel, tailMask);
            AscendC::MicroAPI::Mul(vregInputFp32, vregInputFp32, vregScalePerToken, tailMask);
            AscendC::MicroAPI::StoreAlign<O, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(yAddr + dLoops * floatRepSize + rowOffset, vregInputFp32, tailMask);
            rowOffset += stride;
            scaleOffset += fp32BlockElementNum;
        }
    }
}

/**
* @brief DequantVf 对输入做per-token叠加per-channel的反量化， INT32 ---> FP32.
* @param outputLocal 输出tensor [row, col]
* @param inputLocal 输入tensor [row, col]
* @param scalePerChannelLocal 输入tensor [1, col]
* @param scalePerTokenLocal 输入tensor [row, 8]
* @param row 待处理的行数
* @param col 待处理的列数
* @param stride 待处理数据一行的真实长度
*/
template<typename T, typename C, typename O>
__aicore__ inline void DequantVf(const LocalTensor<O>& outputLocal, const LocalTensor<T>& inputLocal, const LocalTensor<C>& scalePerChannelLocal,
    const LocalTensor<C>& scalePerTokenLocal, uint32_t row, uint32_t col, uint32_t stride)
{
    __ubuf__ O * outputUb = (__ubuf__ O *)outputLocal.GetPhyAddr();
    __ubuf__ T * inputUb = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ C * scalePerChannelLocalUb = (__ubuf__ C *)scalePerChannelLocal.GetPhyAddr();
    __ubuf__ C * scalePerTokenLocalUb = (__ubuf__ C *)scalePerTokenLocal.GetPhyAddr();

    const uint32_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    const uint32_t fp32BlockElementNum = 8;
    uint32_t dLoops = col / floatRepSize;
    uint32_t dTail = col % floatRepSize;
    uint32_t dTailLoop = dTail > 0 ? 1 : 0;
    DequantVFImpl(outputUb, inputUb, scalePerChannelLocalUb, scalePerTokenLocalUb, floatRepSize, fp32BlockElementNum, dLoops, dTail, dTailLoop, row, col, stride);
}
}
#endif
