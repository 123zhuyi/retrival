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
 * \file vf_flashupdate.h
 * \brief
 */
#ifndef VF_FLASH_UPDATE_H
#define VF_FLASH_UPDATE_H

#include "kernel_tensor.h"

namespace AscendC {
/* **************************************************************************************************
 * FlashUpdate
 * [s1, k] = [64, 128], fp32
 * ************************************************************************************************* */

/*
 * @ingroup FlashUpdateTail_VF
 * @brief compute, dstTensor = (preTensor + curTensor) * expMaxTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 64 aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const uint16_t m, const uint16_t d, float dequantScale2)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vDstRegAdd;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrcCur;

        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegTailD = MicroAPI::UpdateMask<T>(pltTailD);

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegAdd, maskRegAll);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                    vDstRegAdd, maskRegTailD);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + j * floatRepSize);
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                    }
                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + j * floatRepSize,
                        vDstRegAdd, maskRegAll);
                }
                // 尾块处理
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + dLoops * floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + dLoops * floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + dLoops * floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + dLoops * floatRepSize,
                    vDstRegAdd, maskRegTailD);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateLastTail_VF
 * @brief compute, dstTensor = (preTensor  + curTensor ) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m, const uint16_t d, float dequantScale2)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vSrcRegAdd;
        MicroAPI::RegTensor<T> vDstRegDiv;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrcCur;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegTailD = MicroAPI::UpdateMask<T>(pltTailD);

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv, maskRegAll);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegTailD);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                    vDstRegDiv, maskRegTailD);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + j * floatRepSize);
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                    }
                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + j * floatRepSize,
                        vDstRegDiv, maskRegAll);
                }
                // 尾块处理
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + dLoops * floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + dLoops * floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + dLoops * floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegTailD);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + dLoops * floatRepSize,
                    vDstRegDiv, maskRegTailD);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateDivTail_VF
 * @brief compute, dstTensor = preTensor / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& preTensor,
    const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d, float dequantScale2)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * preUb = (__ubuf__ MMOUTPUT_T*)preTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vDstRegDiv;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrc;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegTailD = MicroAPI::UpdateMask<T>(pltTailD);

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv, maskRegAll);
                
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                    vDstRegDiv, maskRegTailD);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                    }
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + j * floatRepSize,
                        vDstRegDiv, maskRegAll);
                }
                // 尾块处理
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize + dLoops * floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + dLoops * floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + dLoops * floatRepSize,
                    vDstRegDiv, maskRegTailD);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateNoTail_VF
 * @brief compute, dstTensor = (preTensor + curTensor) * expMaxTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 64 aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateNoTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const uint16_t m, const uint16_t d, float dequantScale2)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vDstRegAdd;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrcCur;

        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegAdd, maskRegAll);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                    vDstRegAdd, maskRegAll);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + j * floatRepSize);
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                    }
                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + j * floatRepSize,
                        vDstRegAdd, maskRegAll);
                }
            }
        }
    }
}

/*
 * @ingroup FlashUpdateLastNoTail_VF
 * @brief compute, dstTensor = (preTensor  + curTensor ) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastNoTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m, const uint16_t d, float dequantScale2)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vSrcRegAdd;
        MicroAPI::RegTensor<T> vDstRegDiv;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrcCur;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize ==128) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv, maskRegAll);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                    vDstRegDiv, maskRegAll);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + j * floatRepSize);
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2, maskRegAll);
                    }
                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + j * floatRepSize,
                        vDstRegDiv, maskRegAll);
                }
            }
        }
    }
}

/*
 * @ingroup FlashUpdateDivNoTail_VF
 * @brief compute, dstTensor = preTensor / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivNoTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& preTensor,
    const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d, float dequantScale2)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * preUb = (__ubuf__ MMOUTPUT_T*)preTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vDstRegDiv;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrc;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv, maskRegAll);
                
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                    vDstRegDiv, maskRegAll);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2, maskRegAll);
                    }
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + j * floatRepSize,
                        vDstRegDiv, maskRegAll);
                }
            }
        }
    }
}

template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdate(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const uint16_t m, const uint16_t d, float dequantScale2)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdate, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if (d % floatRepSize == 0) {
        FlashUpdateNoTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor, expMaxTensor, m, d, dequantScale2);
    } else {
        FlashUpdateTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor, expMaxTensor, m, d, dequantScale2);
    }
}

template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLast(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d, float dequantScale2)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if (d % floatRepSize == 0) {
        FlashUpdateLastNoTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, dequantScale2);
    } else {
        FlashUpdateLastTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, dequantScale2);
    }
}

template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDiv(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& preTensor,
    const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d, float dequantScale2)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateDiv, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if (d % floatRepSize == 0) {
        FlashUpdateDivNoTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, preTensor, expSumTensor, m, d, dequantScale2);
    } else {
        FlashUpdateDivTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, preTensor, expSumTensor, m, d, dequantScale2);
    }
}

template <typename T>
__aicore__ inline void ComputeLseOutput_VF(const LocalTensor<T>& dstTensor, const LocalTensor<T>& softmaxSumTensor,
    const LocalTensor<T>& softmaxMaxTensor, uint32_t dealCount)
{
    __ubuf__ T * srcSumUb = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
    __ubuf__ T * srcMaxUb = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();
    __ubuf__ T * dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregSum;
        MicroAPI::RegTensor<T> vregMax;
        MicroAPI::RegTensor<T> vregRes;
        MicroAPI::RegTensor<T> vregResFinal;
        MicroAPI::RegTensor<float> vregMinValue;
        MicroAPI::RegTensor<float> vregInfValue;
        MicroAPI::MaskReg pregCompare;
        constexpr uint32_t dealRows = 8;
        constexpr uint32_t  floatRepSize = 64; // 64: 一个寄存器存64个float
        constexpr float infValue = 3e+99; // 3e+99 for float inf
        constexpr uint32_t tmpMin = 0xFF7FFFFF;
        float minValue = *((float*)&tmpMin);
        uint16_t updateLoops = ((dealCount) + (dealRows - 1)) / dealRows;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
        MicroAPI::Duplicate<float, float>(vregInfValue, infValue);

        for (uint16_t i = 0; i < updateLoops; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregSum, srcSumUb + (i * dealRows));
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregMax, srcMaxUb + (i * dealRows));

            MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSum, pregAll);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregAll);

            MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregAll);
            MicroAPI::Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);

            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + (i * floatRepSize), vregResFinal, pregAll);
        }
    }
}

} // namespace

#endif // VF_FLASH_UPDATE_H
