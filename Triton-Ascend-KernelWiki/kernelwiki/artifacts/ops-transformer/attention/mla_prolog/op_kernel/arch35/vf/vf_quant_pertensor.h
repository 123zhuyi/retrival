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
 * \file vf_quant_pertensor.h
 * \brief 
 */

 #ifndef VF_QUANT_PERTENSOR_H
 #define VF_QUANT_PERTENSOR_H
 #include "kernel_tensor.h"

namespace MlaProlog{
template <typename T, typename U>
__simd_vf__ void QuantPerTensorVFImpl(__ubuf__ T * inputBuf, __ubuf__ T * quantScaleBuf, __ubuf__ U * outputBuf, 
                                        uint32_t cnt, const uint16_t floatRepSize, uint16_t repeatTimes)
{
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

    // float -> fp8e4m3 类型转换模式结构体
    static constexpr MicroAPI::CastTrait CAST_TRAIT_FP32_TO_FP8E4M3 = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    for(uint16_t i = 0; i < uint16_t(repeatTimes); i++) {
        MicroAPI::RegTensor<T> vregSrc;
        MicroAPI::RegTensor<T> vregQuantScale;
        MicroAPI::RegTensor<T> vregResFloat;
        MicroAPI::RegTensor<U> vregRes;
        uint16_t loopOffset = i * floatRepSize;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, inputBuf + loopOffset);
        // 量化系数broadcast到寄存器所有位置
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregQuantScale, quantScaleBuf);

        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregResFloat, vregSrc, vregQuantScale, pregAll);
        
        MicroAPI::Cast<U, float, CAST_TRAIT_FP32_TO_FP8E4M3>(vregRes, vregResFloat, pregAll);
        MicroAPI::StoreAlign<U, MicroAPI::StoreDist::DIST_PACK4_B32>(outputBuf + loopOffset, vregRes, pregAll);   
    }
}

/**
 * @brief QuantPerTensorVF 对一行进行mul,并量化到fp8e4m3 T float U fp8e4m3 可根据不同量化结果扩展
 * @param outputLocal 输出tensor [row, col]，row为rmsnorm输出的结果，均为1
 * @param inputLocal 输入tensor [row, col]
 * @param quantScaleLocal 量化参数tensor [row, 1]
 * @param row 处理数据的行数 默认为1 后续可扩展
 * @param col 处理数据的列数
 */
template <typename T, typename U>
__aicore__ inline void QuantPerTensorVF(const LocalTensor<U> &outputLocal, const LocalTensor<T> &inputLocal, const LocalTensor<T> &quantScaleLocal,
                                  const uint32_t row, const uint32_t col) {
    uint32_t cnt = row * col;
    const uint16_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    uint16_t repeatTimes = (cnt + floatRepSize - 1) / floatRepSize; // 对尾块处理的扩展，循环处理的次数

    __ubuf__ T * inputBuf = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ T * quantScaleBuf = (__ubuf__ T *)quantScaleLocal.GetPhyAddr();
    __ubuf__ U * outputBuf = (__ubuf__ U *)outputLocal.GetPhyAddr();

    QuantPerTensorVFImpl<T, U>(inputBuf, quantScaleBuf, outputBuf, cnt, floatRepSize, repeatTimes);
}
} // namespace MlaProlog
#endif