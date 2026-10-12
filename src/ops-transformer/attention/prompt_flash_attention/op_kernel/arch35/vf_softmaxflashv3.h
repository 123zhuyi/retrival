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
 * \file vf_softmaxflashv3.h
 * \brief
 */
#ifndef VF_SOFTMAX_FLASH_V3_H
#define VF_SOFTMAX_FLASH_V3_H

#include "kernel_tensor.h"

namespace AscendC {
/* **************************************************************************************************
 * SoftmaxFlashV3
 * ************************************************************************************************* */

// 64 < originN < 128, No update
template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3NoUpdateGeneralImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    constexpr uint32_t floatRepSize = 64;
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint32_t blockU8 = 32;
    constexpr uint32_t blockB16 = 16;
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ float * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * srcUb = (__ubuf__ MMOUTPUT_T*)inSrcTensor.GetPhyAddr();
    __ubuf__ T * srcUbTmp = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)inPseTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUbUnroll = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize].GetPhyAddr();
    __ubuf__ uint8_t * maskBandUb;
    __ubuf__ uint8_t * maskBandUbUnroll;
    if constexpr (isBand) {
        maskBandUb = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner].GetPhyAddr();
        maskBandUbUnroll = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize].GetPhyAddr();
    }

    const uint32_t nPadding = (originN + blockU8 - 1) / blockU8 * blockU8;
    const uint32_t psePadding = (originN + blockB16 - 1) / blockB16 * blockB16;
    const uint32_t oriTailN = originN - floatRepSize;
    const uint32_t tailN = sInner - floatRepSize;
    uint32_t pltOriTailN = oriTailN;
    uint32_t pltTailN = tailN;
    uint32_t pltN = sInner;
    uint32_t pltOriTailN2 = oriTailN * (sizeof(T) / sizeof(pseShiftType));

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregSel;
        MicroAPI::RegTensor<T> vregSelUnroll;
        MicroAPI::RegTensor<T> vregSelBand;
        MicroAPI::RegTensor<T> vregSelBandUnroll;
        MicroAPI::RegTensor<T> vregInputX;
        MicroAPI::RegTensor<T> vregInputXUnroll;
        MicroAPI::RegTensor<T> vregInputXUnrollNew;
        MicroAPI::RegTensor<T> vregMaxTmp;
        MicroAPI::RegTensor<T> vregInputMax;
        MicroAPI::RegTensor<T> vregMaxBrc;
        MicroAPI::RegTensor<T> vregExpSum;
        MicroAPI::RegTensor<T> vregExpEven;
        MicroAPI::RegTensor<T> vregExpOdd;
        // pse
        MicroAPI::RegTensor<pseShiftType> vregPseShift;
        MicroAPI::RegTensor<pseShiftType> vregPseShiftUnroll;
        MicroAPI::RegTensor<T> vregPseShiftCast;
        MicroAPI::RegTensor<T> vregPseShiftCastUnroll;

        MicroAPI::RegTensor<T2> vregExpEvenF16;
        MicroAPI::RegTensor<T2> vregExpOddF16;
        MicroAPI::RegTensor<T2> vregExpF16;

        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllB16 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllPse = MicroAPI::CreateMask<pseShiftType, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregNB16 = MicroAPI::UpdateMask<T2>(pltN);
        MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(pltTailN);
        MicroAPI::MaskReg pregOriTailN = MicroAPI::UpdateMask<T>(pltOriTailN);
        MicroAPI::MaskReg pregOriTailNPse = MicroAPI::UpdateMask<pseShiftType>(pltOriTailN2);
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompareUnroll;
        MicroAPI::MaskReg pregCompareBand;
        MicroAPI::MaskReg pregCompareBandUnroll;

        // int8/fp8 quant
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant;
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuantUnroll;
        MicroAPI::RegTensor<half> vregCastB16;
        MicroAPI::RegTensor<half> vregCastB16Unroll;
        MicroAPI::RegTensor<half> vregCastRes;
        MicroAPI::RegTensor<half> vregMulsRes;
        MicroAPI::RegTensor<T> vregMulsEven;
        MicroAPI::RegTensor<T> vregMulsOdd;
        MicroAPI::RegTensor<T2> vregCast;
        MicroAPI::RegTensor<T2> vregRes;
        MicroAPI::MaskReg pregAllFp16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate<T, T>(vregMin, minValue);
        if constexpr (hasAtten == 1 && isMlaSGD) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (pregCompare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (pregCompareUnroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
        }
        for (uint16_t i = 0; i < rows; ++i) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant, srcUb + i * sInner);
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuantUnroll, srcUb + floatRepSize + i * sInner);

                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX, vregInputQuant, pregAll);
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputXUnroll, vregInputQuantUnroll, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputX, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputXUnroll, srcUb + floatRepSize + i * sInner);
            }

            if constexpr (IsSameType<T2, int8_t>::value ||
                IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, dequantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll, dequantScale1, pregAll);
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputX, vregInputX, scale, pregAll);  // Muls(scale)
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputXUnroll, vregInputXUnroll, scale, pregOriTailN);
            MicroAPI::Select<T>(vregInputXUnroll, vregInputXUnroll, vregMin, pregOriTailN); // 筛除actual seq length引入的脏数据

            if constexpr (hasPse == 1) {
                // fp16/bf16 -> fp32
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift, pseUb + i * psePadding);
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShiftUnroll, pseUb + floatRepSize + i * psePadding);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                    MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast, vregPseShift, pregAllPse);
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCastUnroll, vregPseShiftUnroll, pregOriTailNPse);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, vregPseShiftCast, pregAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll, vregPseShiftCastUnroll, pregOriTailN); // Avoid access out of range
                MicroAPI::Select<T>(vregInputXUnroll, vregInputXUnroll, vregMin, pregOriTailN);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSGD) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare, ((__ubuf__ uint32_t*)(maskUb + i * nPadding)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareUnroll, ((__ubuf__ uint32_t*)(maskUbUnroll + i * nPadding)));
                }
                if constexpr (isBand) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand, ((__ubuf__ uint32_t*)(maskBandUb + i * nPadding)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBandUnroll, ((__ubuf__ uint32_t*)(maskBandUbUnroll + i * nPadding)));
                    MicroAPI::Select<T>(vregSelBand, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelBandUnroll, vregMin, vregInputXUnroll, pregCompareUnroll);
                    MicroAPI::Select<T>(vregSel, vregSelBand, vregMin, pregCompareBand);
                    MicroAPI::Select<T>(vregSelUnroll, vregSelBandUnroll, vregMin, pregCompareBandUnroll);
                } else {
                    MicroAPI::Select<T>(vregSel, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelUnroll, vregMin, vregInputXUnroll, pregCompareUnroll);
                }
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregSel, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregSelUnroll, pregTailN);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregSel, vregSelUnroll, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp, pregAll);
            } else {
                MicroAPI::Select<T>(vregInputXUnrollNew, vregInputXUnroll, vregMin, pregOriTailN);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregInputX, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregInputXUnrollNew, pregTailN);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregInputX, vregInputXUnrollNew, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp, pregAll);
            }
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (maxUb, vregInputMax, uregMax, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (maxUb, uregMax, 0);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>
                (vregMaxBrc, maxUbStart + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vregInputX, vregInputXUnroll, srcUbTmp + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpEven, vregInputX, vregMaxBrc, pregAll);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpOdd, vregInputXUnroll, vregMaxBrc, pregAll);

            if constexpr (IsSameType<T2, int8_t>::value) {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<half, T, castTrait0>(vregCastB16, vregExpEven, pregAll);
                MicroAPI::Cast<half, T, castTrait1>(vregCastB16Unroll, vregExpOdd, pregAll);

                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregCastRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCastB16, (MicroAPI::RegTensor<uint16_t>&)vregCastB16Unroll,
                    pregAllFp16);
                MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes, vregCastRes, (half)quantScale1, pregAllFp16);

                MicroAPI::Cast<T2, half, castTrait0>(vregCast, vregMulsRes, pregAllFp16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vregRes, blockStride, repeatStride, pregNB16);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsEven, vregExpEven, quantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsOdd, vregExpOdd, quantScale1, pregAll);
                if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
                } else {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
                }
                MicroAPI::Or<uint8_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint8_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint8_t>&)vregExpEvenF16, (MicroAPI::RegTensor<uint8_t>&)vregExpOddF16,
                    pregAllB16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb,
                    vregRes, blockStride, repeatStride, pregNB16);
            } else {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregExpEven, pregAll);
                MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregExpOdd, pregAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>
                    ((MicroAPI::RegTensor<uint16_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpEvenF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpOddF16, pregAllB16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb, vregExpF16, blockStride,
                    repeatStride, pregNB16);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpEven, vregExpOdd, pregAll);
            MicroAPI::ReduceSum<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpSum, pregAll);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (expSumUb, vregExpSum, uregExpSum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (expSumUb, uregExpSum, 0);
    }
}

// originN = 128, No update
template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3NoUpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    constexpr uint32_t floatRepSize = 64;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ float * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * srcUb = (__ubuf__ MMOUTPUT_T*)inSrcTensor.GetPhyAddr();
    __ubuf__ T * srcUbTmp = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)inPseTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUbUnroll = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize].GetPhyAddr();
    __ubuf__ uint8_t * maskBandUb;
    __ubuf__ uint8_t * maskBandUbUnroll;
    if constexpr (isBand) {
        maskBandUb = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner].GetPhyAddr();
        maskBandUbUnroll = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize].GetPhyAddr();
    }
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregSel;
        MicroAPI::RegTensor<T> vregSelUnroll;
        MicroAPI::RegTensor<T> vregSelBand;
        MicroAPI::RegTensor<T> vregSelBandUnroll;
        MicroAPI::RegTensor<T> vregInputX;
        MicroAPI::RegTensor<T> vregInputXUnroll;
        MicroAPI::RegTensor<T> vregMaxTmp;
        MicroAPI::RegTensor<T> vregInputMax;
        MicroAPI::RegTensor<T> vregMaxBrc;
        MicroAPI::RegTensor<T> vregExpSum;
        MicroAPI::RegTensor<T> vregExpEven;
        MicroAPI::RegTensor<T> vregExpOdd;
        // half / bfloat16_t
        MicroAPI::RegTensor<T2> vregExpEvenF16;
        MicroAPI::RegTensor<T2> vregExpOddF16;
        MicroAPI::RegTensor<T2> vregExpF16;
        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;
        // pse
        MicroAPI::RegTensor<pseShiftType> vregPseShift;
        MicroAPI::RegTensor<pseShiftType> vregPseShiftUnroll;
        MicroAPI::RegTensor<T> vregPseShiftCast;
        MicroAPI::RegTensor<T> vregPseShiftCastUnroll;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllB16 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllPse = MicroAPI::CreateMask<pseShiftType, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompareUnroll;
        MicroAPI::MaskReg pregCompareBand;
        MicroAPI::MaskReg pregCompareBandUnroll;

        // int8/fp8 quant
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant;
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuantUnroll;
        MicroAPI::RegTensor<half> vregCastB16;
        MicroAPI::RegTensor<half> vregCastB16Unroll;
        MicroAPI::RegTensor<half> vregCastRes;
        MicroAPI::RegTensor<half> vregMulsRes;
        MicroAPI::RegTensor<T> vregMulsEven;
        MicroAPI::RegTensor<T> vregMulsOdd;
        MicroAPI::RegTensor<T2> vregCast;
        MicroAPI::RegTensor<T2> vregRes;
        MicroAPI::MaskReg pregAllFp16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregS8 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::VL128>();

        if constexpr (hasAtten == 1) {
            MicroAPI::Duplicate<T, T>(vregMin, minValue);
            if constexpr (isMlaSGD) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (pregCompare, ((__ubuf__ uint32_t*)(maskUb)));
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (pregCompareUnroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
            }
        }
        for (uint16_t i = 0; i < rows; ++i) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant, srcUb + i * sInner);
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuantUnroll, srcUb + floatRepSize + i * sInner);

                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX, vregInputQuant, pregAll);
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputXUnroll, vregInputQuantUnroll, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputX, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputXUnroll, srcUb + floatRepSize + i * sInner);
            }

            if constexpr (IsSameType<T2, int8_t>::value ||
                IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, dequantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll, dequantScale1, pregAll);
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX,
                scale, pregAll); // Muls(scale)
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll,
                scale, pregAll);

            if constexpr (hasPse == 1) {
                // fp16/bf16 -> fp32
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift, pseUb + i * sInner);
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShiftUnroll, pseUb + floatRepSize + i * sInner);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                    MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast, vregPseShift, pregAllPse);
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCastUnroll, vregPseShiftUnroll, pregAllPse);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, vregPseShiftCast, pregAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll, vregPseShiftCastUnroll, pregAll);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSGD) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare, ((__ubuf__ uint32_t*)(maskUb + i * sInner)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareUnroll, ((__ubuf__ uint32_t*)(maskUbUnroll + i * sInner)));
                }
                if constexpr (isBand) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand, ((__ubuf__ uint32_t*)(maskBandUb + i * sInner)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBandUnroll, ((__ubuf__ uint32_t*)(maskBandUbUnroll + i * sInner)));
                    MicroAPI::Select<T>(vregSelBand, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelBandUnroll, vregMin, vregInputXUnroll, pregCompareUnroll);
                    MicroAPI::Select<T>(vregSel, vregSelBand, vregMin, pregCompareBand);
                    MicroAPI::Select<T>(vregSelUnroll, vregSelBandUnroll, vregMin, pregCompareBandUnroll);
                } else {
                    MicroAPI::Select<T>(vregSel, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelUnroll, vregMin, vregInputXUnroll, pregCompareUnroll);
                }
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregSel, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregSelUnroll, pregAll);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregSel, vregSelUnroll, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregInputX, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregInputXUnroll, pregAll);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregInputX, vregInputXUnroll, pregAll);
            }
            MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputMax, vregMaxTmp, pregAll);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (maxUb, vregInputMax, uregMax, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (maxUb, uregMax, 0);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>
                (vregMaxBrc, maxUbStart + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vregInputX, vregInputXUnroll, srcUbTmp + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpEven, vregInputX, vregMaxBrc, pregAll);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpOdd, vregInputXUnroll, vregMaxBrc, pregAll);

            if constexpr (IsSameType<T2, int8_t>::value) {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<half, T, castTrait0>(vregCastB16, vregExpEven, pregAll);
                MicroAPI::Cast<half, T, castTrait1>(vregCastB16Unroll, vregExpOdd, pregAll);

                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregCastRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCastB16, (MicroAPI::RegTensor<uint16_t>&)vregCastB16Unroll,
                    pregAllFp16);
                MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes, vregCastRes, (half)quantScale1, pregAllFp16);

                MicroAPI::Cast<T2, half, castTrait0>(vregCast, vregMulsRes, pregAllFp16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vregRes, blockStride, repeatStride, pregS8);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsEven, vregExpEven, quantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsOdd, vregExpOdd, quantScale1, pregAll);
                if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
                } else {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
                }
                MicroAPI::Or<uint8_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint8_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint8_t>&)vregExpEvenF16, (MicroAPI::RegTensor<uint8_t>&)vregExpOddF16,
                    pregAllB16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb,
                    vregRes, blockStride, repeatStride, pregS8);
            } else {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregExpEven, pregAll);
                MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregExpOdd, pregAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>
                    ((MicroAPI::RegTensor<uint16_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpEvenF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpOddF16, pregAllB16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb, vregExpF16, blockStride,
                    repeatStride, pregAllB16);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpEven, vregExpOdd, pregAll);
            MicroAPI::ReduceSum<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpSum, pregAll);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (expSumUb, vregExpSum, uregExpSum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (expSumUb, uregExpSum, 0);
    }
}

// originN <= 64, No update
template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3NoUpdateImpl64(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ float * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * srcUb = (__ubuf__ MMOUTPUT_T*)inSrcTensor.GetPhyAddr();
    __ubuf__ T * srcUbTmp = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)inPseTensor.GetPhyAddr();

    constexpr uint32_t floatRepSize = 64;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint32_t blockU8 = 32;
    constexpr uint32_t blockB16 = 16;
    constexpr uint16_t repeatStride = 1;

    __ubuf__ uint8_t * maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskBandUb;
    if constexpr (isBand) {
        maskBandUb = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner].GetPhyAddr();
    }

    const uint32_t nPadding = (originN + blockU8 - 1) / blockU8 * blockU8;
    const uint32_t psePadding = (originN + blockB16 - 1) / blockB16 * blockB16;
    uint32_t pltOriginalN = originN;
    uint32_t pltOriginalN2 = originN * (sizeof(T) / sizeof(pseShiftType));
    uint32_t pltSrcN = sInner;
    uint32_t pltSrcN16 = sInner;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregSel;
        MicroAPI::RegTensor<T> vregSelBand;
        MicroAPI::RegTensor<T> vregInputX;
        MicroAPI::RegTensor<T> vregInputMax;
        MicroAPI::RegTensor<T> vregMaxBrc;
        MicroAPI::RegTensor<T> vregExp;
        MicroAPI::RegTensor<T> vregExpSum;
        // half / bfloat16_t
        MicroAPI::RegTensor<T2> vregExpEvenF16;
        MicroAPI::RegTensor<T2> vregExpF16;
        MicroAPI::RegTensor<T2> vregDstEvenF16;
        MicroAPI::RegTensor<T2> vregDstOddF16;
        MicroAPI::RegTensor<T2> vregDummyF16;
        // pse
        MicroAPI::RegTensor<pseShiftType> vregPseShift;
        MicroAPI::RegTensor<T> vregPseShiftCast;

        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllB16 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregSrcN = MicroAPI::UpdateMask<T>(pltSrcN);
        MicroAPI::MaskReg pregSrcNB16 = MicroAPI::UpdateMask<T2>(pltSrcN16);
        MicroAPI::MaskReg pregOriSrcN = MicroAPI::UpdateMask<T>(pltOriginalN);
        MicroAPI::MaskReg pregOriSrcNPse = MicroAPI::UpdateMask<pseShiftType>(pltOriginalN2);
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompareBand;

        // int8/fp8 quant
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant;
        MicroAPI::RegTensor<half> vregCastB16;
        MicroAPI::RegTensor<half> vregCastRes;
        MicroAPI::RegTensor<half> vregMulsRes;
        MicroAPI::RegTensor<T> vregMulsB32;
        MicroAPI::RegTensor<T2> vregCast;
        MicroAPI::RegTensor<T2> vregRes;
        MicroAPI::MaskReg pregAllFp16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();

        if constexpr (hasAtten == 1) {
            MicroAPI::Duplicate<T, T>(vregMin, minValue);
            if constexpr (isMlaSGD) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (pregCompare, ((__ubuf__ uint32_t*)(maskUb)));
            }
        }

        // x_max = max(src, axis=-1, keepdims=True)
        for (uint16_t i = 0; i < rows; ++i) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant, srcUb + i * sInner);

                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX, vregInputQuant, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputX, srcUb + i * sInner);
            }

            if constexpr (IsSameType<T2, int8_t>::value ||
                IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, dequantScale1, pregAll);
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputX, vregInputX, scale, pregOriSrcN);  // Muls(scale)
            MicroAPI::Select<T>(vregInputX, vregInputX, vregMin, pregOriSrcN); // 筛除actual seq length引入的脏数据

            if constexpr (hasPse == 1) {
                // fp16/bf16 -> fp32
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift, pseUb + i * psePadding);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                    MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast, vregPseShift, pregOriSrcNPse); // Avoid access out of range
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, vregPseShiftCast, pregOriSrcN);
                MicroAPI::Select<T>(vregInputX, vregInputX, vregMin, pregOriSrcN);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSGD) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare, ((__ubuf__ uint32_t*)(maskUb + i * nPadding)));
                }
                if constexpr (isBand) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand, ((__ubuf__ uint32_t*)(maskBandUb + i * nPadding)));
                    MicroAPI::Select<T>(vregSelBand, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSel, vregSelBand, vregMin, pregCompareBand);
                } else {
                    MicroAPI::Select<T>(vregSel, vregMin, vregInputX, pregCompare);
                }
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregSel, pregSrcN);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregSel, pregOriSrcN);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregInputX, pregSrcN);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregInputX, pregOriSrcN);
            }
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (maxUb, vregInputMax, uregMax, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (maxUb, uregMax, 0);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>
                (vregMaxBrc, maxUbStart + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                (vregInputX, srcUbTmp + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExp, vregInputX, vregMaxBrc, pregOriSrcN);

            if constexpr (IsSameType<T2, int8_t>::value) {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<half, T, castTrait0>(vregCastB16, vregExp, pregAll);

                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)vregCastRes,
                    (MicroAPI::RegTensor<uint32_t>&)vregCastB16);
                MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes, vregCastRes, (half)quantScale1, pregAllFp16);

                MicroAPI::Cast<T2, half, castTrait0>(vregCast, vregMulsRes, pregAllFp16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vregRes, blockStride, repeatStride, pregSrcNB16);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsB32, vregExp, quantScale1, pregAll);
                if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsB32, pregAll);
                } else {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsB32, pregAll);
                }
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint32_t>&)vregExpEvenF16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb,
                    vregRes, blockStride, repeatStride, pregSrcNB16);
            } else {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T2, T, castTrait0>(vregExpF16, vregExp, pregAllB16);
                MicroAPI::DeInterleave<uint16_t>((MicroAPI::RegTensor<uint16_t> &)vregDstEvenF16,
                    (MicroAPI::RegTensor<uint16_t> &)vregDstOddF16,
                    (MicroAPI::RegTensor<uint16_t> &)vregExpF16,
                    (MicroAPI::RegTensor<uint16_t> &)vregDummyF16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb, vregDstEvenF16, blockStride,
                    repeatStride, pregSrcNB16);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::ReduceSum<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExp, pregOriSrcN);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (expSumUb, vregExpSum, uregExpSum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (expSumUb, uregExpSum, 0);
    }
}

// 128 < originN <= 256, No update
template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3NoUpdateGeneralImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    constexpr uint32_t floatRepSize = 64;
    constexpr uint32_t floatRepSize2 = 128;
    constexpr uint32_t floatRepSize3 = 192;
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint32_t blockU8 = 32;
    constexpr uint32_t blockB16 = 16;
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb1 = ((__ubuf__ T2*)dstTensor.GetPhyAddr()) + (vsOuter + 1) * (sInner >> 1U);
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ float * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * srcUb = (__ubuf__ MMOUTPUT_T*)inSrcTensor.GetPhyAddr();
    __ubuf__ T * srcUbTmp = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)inPseTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb1 = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize].GetPhyAddr();
	__ubuf__ uint8_t * maskUb2 = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize2].GetPhyAddr();
	__ubuf__ uint8_t * maskUb3 = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize3].GetPhyAddr();
    __ubuf__ uint8_t * maskBandUb;
    __ubuf__ uint8_t * maskBandUb1;
	__ubuf__ uint8_t * maskBandUb2;
	__ubuf__ uint8_t * maskBandUb3;
    if constexpr (isBand) {
        maskBandUb = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner].GetPhyAddr();
        maskBandUb1 = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize].GetPhyAddr();
		maskBandUb2 = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize2].GetPhyAddr();
		maskBandUb3 = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize3].GetPhyAddr();
    }

    const uint32_t nPadding = (originN + blockU8 - 1) / blockU8 * blockU8;
    const uint32_t psePadding = (originN + blockB16 - 1) / blockB16 * blockB16;
    const uint32_t oriTailN1 = (originN - floatRepSize2 < floatRepSize) ? (originN - floatRepSize2) : floatRepSize;
	const uint32_t oriTailN2 = (originN <= floatRepSize3) ? 0 : (originN - floatRepSize3);
    const uint32_t tailN1 = sInner - floatRepSize2;
	const uint32_t tailN2 = sInner - floatRepSize3;
    uint32_t pltOriTailN1 = oriTailN1;
	uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltTailN1 = tailN1;
	uint32_t pltTailN2 = tailN2;
    uint32_t pltN = sInner;
    uint32_t pltOriTailPse1 = oriTailN1 * (sizeof(T) / sizeof(pseShiftType));
	uint32_t pltOriTailPse2 = oriTailN2 * (sizeof(T) / sizeof(pseShiftType));

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregSel;
        MicroAPI::RegTensor<T> vregSel1;
        MicroAPI::RegTensor<T> vregSel2;
        MicroAPI::RegTensor<T> vregSel3;
        MicroAPI::RegTensor<T> vregSelBand;
        MicroAPI::RegTensor<T> vregSelBand1;
		MicroAPI::RegTensor<T> vregSelBand2;
		MicroAPI::RegTensor<T> vregSelBand3;
        MicroAPI::RegTensor<T> vregInputX;
        MicroAPI::RegTensor<T> vregInputX1;
		MicroAPI::RegTensor<T> vregInputX2;
		MicroAPI::RegTensor<T> vregInputX3;
        MicroAPI::RegTensor<T> vregInputX2New;
		MicroAPI::RegTensor<T> vregInputX3New;
        MicroAPI::RegTensor<T> vregMaxTmp;
		MicroAPI::RegTensor<T> vregMaxTmp1;
		MicroAPI::RegTensor<T> vregMaxTmp2;
        MicroAPI::RegTensor<T> vregInputMax;
        MicroAPI::RegTensor<T> vregMaxBrc;
        MicroAPI::RegTensor<T> vregExpSum;
		MicroAPI::RegTensor<T> vregExpSum1;
        MicroAPI::RegTensor<T> vregExpEven;
		MicroAPI::RegTensor<T> vregExpEven1;
        MicroAPI::RegTensor<T> vregExpOdd;
		MicroAPI::RegTensor<T> vregExpOdd1;
        // pse
        MicroAPI::RegTensor<pseShiftType> vregPseShift;
        MicroAPI::RegTensor<pseShiftType> vregPseShift1;
		MicroAPI::RegTensor<pseShiftType> vregPseShift2;
		MicroAPI::RegTensor<pseShiftType> vregPseShift3;
        MicroAPI::RegTensor<T> vregPseShiftCast;
        MicroAPI::RegTensor<T> vregPseShiftCast1;
		MicroAPI::RegTensor<T> vregPseShiftCast2;
		MicroAPI::RegTensor<T> vregPseShiftCast3;

        MicroAPI::RegTensor<T2> vregExpEvenF16;
		MicroAPI::RegTensor<T2> vregExpEvenF161;
        MicroAPI::RegTensor<T2> vregExpOddF16;
		MicroAPI::RegTensor<T2> vregExpOddF161;
        MicroAPI::RegTensor<T2> vregExpF16;
		MicroAPI::RegTensor<T2> vregExpF161;

        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllB16 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllPse = MicroAPI::CreateMask<pseShiftType, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregNB16 = MicroAPI::UpdateMask<T2>(pltN);
        MicroAPI::MaskReg pregTailN1 = MicroAPI::UpdateMask<T>(pltTailN1);
		MicroAPI::MaskReg pregTailN2 = MicroAPI::UpdateMask<T>(pltTailN2);
        MicroAPI::MaskReg pregOriTailN1 = MicroAPI::UpdateMask<T>(pltOriTailN1);
		MicroAPI::MaskReg pregOriTailN2 = MicroAPI::UpdateMask<T>(pltOriTailN2);
        MicroAPI::MaskReg pregOriTailNPse1 = MicroAPI::UpdateMask<pseShiftType>(pltOriTailPse1);
		MicroAPI::MaskReg pregOriTailNPse2 = MicroAPI::UpdateMask<pseShiftType>(pltOriTailPse2);
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompare1;
		MicroAPI::MaskReg pregCompare2;
		MicroAPI::MaskReg pregCompare3;
        MicroAPI::MaskReg pregCompareBand;
        MicroAPI::MaskReg pregCompareBand1;
		MicroAPI::MaskReg pregCompareBand2;
		MicroAPI::MaskReg pregCompareBand3;

        // int8/fp8 quant
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant;
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant1;
		MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant2;
		MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant3;
        MicroAPI::RegTensor<half> vregCastB16;
        MicroAPI::RegTensor<half> vregCastB161;
		MicroAPI::RegTensor<half> vregCastB162;
		MicroAPI::RegTensor<half> vregCastB163;
        MicroAPI::RegTensor<half> vregCastRes;
		MicroAPI::RegTensor<half> vregCastRes1;
        MicroAPI::RegTensor<half> vregMulsRes;
		MicroAPI::RegTensor<half> vregMulsRes1;
        MicroAPI::RegTensor<T> vregMulsEven;
		MicroAPI::RegTensor<T> vregMulsEven1;
        MicroAPI::RegTensor<T> vregMulsOdd;
		MicroAPI::RegTensor<T> vregMulsOdd1;
        MicroAPI::RegTensor<T2> vregCast;
		MicroAPI::RegTensor<T2> vregCast1;
        MicroAPI::RegTensor<T2> vregRes;
		MicroAPI::RegTensor<T2> vregRes1;
        MicroAPI::MaskReg pregAllFp16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate<T, T>(vregMin, minValue);
        if constexpr (hasAtten == 1 && isMlaSGD) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>(pregCompare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>(pregCompare1, ((__ubuf__ uint32_t*)(maskUb1)));
			MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>(pregCompare2, ((__ubuf__ uint32_t*)(maskUb2)));
			MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>(pregCompare3, ((__ubuf__ uint32_t*)(maskUb3)));
        }
        for (uint16_t i = 0; i < rows; ++i) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant, srcUb + i * sInner);
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant1, srcUb + floatRepSize + i * sInner);
				MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant2, srcUb + floatRepSize2 + i * sInner);
				MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant3, srcUb + floatRepSize3 + i * sInner);

                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX, vregInputQuant, pregAll);
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX1, vregInputQuant1, pregAll);
				MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX2, vregInputQuant2, pregAll);
				MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX3, vregInputQuant3, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInputX, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInputX1, srcUb + floatRepSize + i * sInner);
				MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInputX2, srcUb + floatRepSize2 + i * sInner);
				MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInputX3, srcUb + floatRepSize3 + i * sInner);
            }

            if constexpr (IsSameType<T2, int8_t>::value ||
                IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, dequantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX1, vregInputX1, dequantScale1, pregAll);
				MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX2, vregInputX2, dequantScale1, pregAll);
				MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX3, vregInputX3, dequantScale1, pregAll);
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, scale, pregAll);  // Muls(scale)
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX1, vregInputX1, scale, pregAll);
			MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX2, vregInputX2, scale, pregOriTailN1);
			MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX3, vregInputX3, scale, pregOriTailN2);
            MicroAPI::Select<T>(vregInputX2, vregInputX2, vregMin, pregOriTailN1); // 筛除actual seq length引入的脏数据
			MicroAPI::Select<T>(vregInputX3, vregInputX3, vregMin, pregOriTailN2); // 筛除actual seq length引入的脏数据

            if constexpr (hasPse == 1) {
                // fp16/bf16 -> fp32
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift, pseUb + i * psePadding);
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift1, pseUb + floatRepSize + i * psePadding);
				MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift2, pseUb + floatRepSize2 + i * psePadding);
				MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift3, pseUb + floatRepSize3 + i * psePadding);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                    MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast, vregPseShift, pregAllPse);
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast1, vregPseShift1, pregAllPse);
				MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast2, vregPseShift2, pregOriTailNPse1);
				MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast3, vregPseShift3, pregOriTailNPse2);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, vregPseShiftCast, pregAll);
				MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX1, vregInputX1, vregPseShiftCast1, pregAll);
				MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX2, vregInputX2, vregPseShiftCast2, pregOriTailN1);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX3, vregInputX3, vregPseShiftCast3, pregOriTailN2); // Avoid access out of range
                MicroAPI::Select<T>(vregInputX2, vregInputX2, vregMin, pregOriTailN1);
				MicroAPI::Select<T>(vregInputX3, vregInputX3, vregMin, pregOriTailN2);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSGD) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare, ((__ubuf__ uint32_t*)(maskUb + i * nPadding)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare1, ((__ubuf__ uint32_t*)(maskUb1 + i * nPadding)));
					MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare2, ((__ubuf__ uint32_t*)(maskUb2 + i * nPadding)));
					MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare3, ((__ubuf__ uint32_t*)(maskUb3 + i * nPadding)));
                }
                if constexpr (isBand) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand, ((__ubuf__ uint32_t*)(maskBandUb + i * nPadding)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand1, ((__ubuf__ uint32_t*)(maskBandUb1 + i * nPadding)));
					MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand2, ((__ubuf__ uint32_t*)(maskBandUb2 + i * nPadding)));
					MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand3, ((__ubuf__ uint32_t*)(maskBandUb3 + i * nPadding)));
                    MicroAPI::Select<T>(vregSelBand, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelBand1, vregMin, vregInputX1, pregCompare1);
					MicroAPI::Select<T>(vregSelBand2, vregMin, vregInputX2, pregCompare2);
					MicroAPI::Select<T>(vregSelBand3, vregMin, vregInputX3, pregCompare3);
                    MicroAPI::Select<T>(vregSel, vregSelBand, vregMin, pregCompareBand);
                    MicroAPI::Select<T>(vregSel1, vregSelBand1, vregMin, pregCompareBand1);
					MicroAPI::Select<T>(vregSel2, vregSelBand2, vregMin, pregCompareBand2);
					MicroAPI::Select<T>(vregSel3, vregSelBand3, vregMin, pregCompareBand3);
                } else {
                    MicroAPI::Select<T>(vregSel, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSel1, vregMin, vregInputX1, pregCompare1);
					MicroAPI::Select<T>(vregSel2, vregMin, vregInputX2, pregCompare2);
					MicroAPI::Select<T>(vregSel3, vregMin, vregInputX3, pregCompare3);
                }
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregSel, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregSel1, pregAll);
				MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize2 + i * sInner, vregSel2, pregTailN1);
				MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize3 + i * sInner, vregSel3, pregTailN2);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregSel, vregSel1, pregAll);
				MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp1, vregSel2, vregSel3, pregAll);
				MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp2, vregMaxTmp, vregMaxTmp1, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp2, pregAll);
            } else {
                MicroAPI::Select<T>(vregInputX2New, vregInputX2, vregMin, pregOriTailN1);
				MicroAPI::Select<T>(vregInputX3New, vregInputX3, vregMin, pregOriTailN2);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregInputX, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregInputX1, pregAll);
				MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize2 + i * sInner, vregInputX2New, pregTailN1);
				MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize3 + i * sInner, vregInputX3New, pregTailN2);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregInputX, vregInputX1, pregAll);
				MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp1, vregInputX2New, vregInputX3New, pregAll);
				MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp2, vregMaxTmp, vregMaxTmp1, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp2, pregAll);
            }
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (maxUb, vregInputMax, uregMax, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (maxUb, uregMax, 0);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>
                (vregMaxBrc, maxUbStart + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vregInputX, vregInputX1, srcUbTmp + i * sInner);
			MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vregInputX2, vregInputX3, srcUbTmp + floatRepSize2 + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpEven, vregInputX, vregMaxBrc, pregAll);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpOdd, vregInputX1, vregMaxBrc, pregAll);
			MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpEven1, vregInputX2, vregMaxBrc, pregAll);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpOdd1, vregInputX3, vregMaxBrc, pregAll);

            if constexpr (IsSameType<T2, int8_t>::value) {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<half, T, castTrait0>(vregCastB16, vregExpEven, pregAll);
                MicroAPI::Cast<half, T, castTrait1>(vregCastB161, vregExpOdd, pregAll);
				MicroAPI::Cast<half, T, castTrait0>(vregCastB162, vregExpEven1, pregAll);
                MicroAPI::Cast<half, T, castTrait1>(vregCastB163, vregExpOdd1, pregAll);

                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregCastRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCastB16, (MicroAPI::RegTensor<uint16_t>&)vregCastB161,
                    pregAllFp16);
				MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregCastRes1,
                    (MicroAPI::RegTensor<uint16_t>&)vregCastB162, (MicroAPI::RegTensor<uint16_t>&)vregCastB163,
                    pregAllFp16);
                MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes, vregCastRes, (half)quantScale1, pregAllFp16);
				MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes1, vregCastRes1, (half)quantScale1, pregAllFp16);

                MicroAPI::Cast<T2, half, castTrait0>(vregCast, vregMulsRes, pregAllFp16);
				MicroAPI::Cast<T2, half, castTrait1>(vregCast1, vregMulsRes1, pregAllFp16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast);
				MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes1,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast1);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vregRes, blockStride, repeatStride, pregNB16);
				MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb1),
                    vregRes1, blockStride, repeatStride, pregNB16);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsEven, vregExpEven, quantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsOdd, vregExpOdd, quantScale1, pregAll);
				MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsEven1, vregExpEven1, quantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsOdd1, vregExpOdd1, quantScale1, pregAll);
                if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
					MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF161, vregMulsEven1, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF161, vregMulsOdd1, pregAll);
                } else {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
					MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF161, vregMulsEven1, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF161, vregMulsOdd1, pregAll);
                }
                MicroAPI::Or<uint8_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint8_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint8_t>&)vregExpEvenF16, (MicroAPI::RegTensor<uint8_t>&)vregExpOddF16,
                    pregAllB16);
				MicroAPI::Or<uint8_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint8_t>&)vregExpF161,
                    (MicroAPI::RegTensor<uint8_t>&)vregExpEvenF161, (MicroAPI::RegTensor<uint8_t>&)vregExpOddF161,
                    pregAllB16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF16);
				MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes1,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF161);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb,
                    vregRes, blockStride, repeatStride, pregNB16);
				MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb1,
                    vregRes1, blockStride, repeatStride, pregNB16);
            } else {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregExpEven, pregAll);
                MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregExpOdd, pregAll);
				MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF161, vregExpEven1, pregAll);
                MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF161, vregExpOdd1, pregAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>
                    ((MicroAPI::RegTensor<uint16_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpEvenF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpOddF16, pregAllB16);
				MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>
                    ((MicroAPI::RegTensor<uint16_t>&)vregExpF161,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpEvenF161,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpOddF161, pregAllB16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb, vregExpF16, blockStride,
                    repeatStride, pregNB16);
				MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb1, vregExpF161, blockStride,
                    repeatStride, pregNB16);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpEven, vregExpOdd, pregAll);
			MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum1, vregExpEven1, vregExpOdd1, pregAll);
			MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpSum, vregExpSum1, pregAll);
            MicroAPI::ReduceSum<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpSum, pregAll);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (expSumUb, vregExpSum, uregExpSum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (expSumUb, uregExpSum, 0);
    }
}

template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3NoUpdate8(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    // mode 1: originN = 128
    if constexpr (mode == 1) {
        SoftmaxFlashV3NoUpdateImpl128<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    } else if constexpr (mode == 2) { // mode 2: 0 < originN <= 64
        SoftmaxFlashV3NoUpdateImpl64<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    } else if constexpr (mode == 0) { // mode 0: 64 < originN < 128
        SoftmaxFlashV3NoUpdateGeneralImpl128<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    } else { // mode 3: 128 < originN <= 256
        SoftmaxFlashV3NoUpdateGeneralImpl256<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    }
}

// 64 < originN < 128, Update
template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3UpdateGeneralImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    constexpr uint32_t floatRepSize = 64;
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint32_t blockU8 = 32;
    constexpr uint32_t blockB16 = 16;
    const uint32_t nPadding = (originN + blockU8 - 1) / blockU8 * blockU8;
    const uint32_t psePadding = (originN + blockB16 - 1) / blockB16 * blockB16;
    constexpr uint16_t repeatStride = 1;

    const uint32_t oriTailN = originN - floatRepSize;
    const uint32_t tailN = sInner - floatRepSize;
    uint32_t pltOriTailN = oriTailN;
    uint32_t pltOriTailN2 = oriTailN * (sizeof(T) / sizeof(pseShiftType));
    uint32_t pltTailN = tailN;
    uint32_t pltN = sInner;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * srcUb = (__ubuf__ MMOUTPUT_T*)inSrcTensor.GetPhyAddr();
    __ubuf__ T * srcUbTmp = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
    __ubuf__ float * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ float * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpExpSumUbStart = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + m * reduceN;
    __ubuf__ float * tmpMaxUbStart = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + m * reduceN;
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)inPseTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUbUnroll = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize].GetPhyAddr();
    __ubuf__ uint8_t * maskBandUb;
    __ubuf__ uint8_t * maskBandUbUnroll;
    if constexpr (isBand) {
        maskBandUb = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner].GetPhyAddr();
        maskBandUbUnroll = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize].GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregSel;
        MicroAPI::RegTensor<T> vregSelUnroll;
        MicroAPI::RegTensor<T> vregSelBand;
        MicroAPI::RegTensor<T> vregSelBandUnroll;
        MicroAPI::RegTensor<T> vregInputX;
        MicroAPI::RegTensor<T> vregInputXUnroll;
        MicroAPI::RegTensor<T> vregInputXUnrollNew;
        MicroAPI::RegTensor<T> vregMaxTmp;
        MicroAPI::RegTensor<T> vregInputMax;
        MicroAPI::RegTensor<T> vregMaxBrc;
        MicroAPI::RegTensor<T> vregExpSum;
        MicroAPI::RegTensor<T> vregExpSumBrc;
        MicroAPI::RegTensor<T> vregInMax;
        MicroAPI::RegTensor<T> vregMax;
        MicroAPI::RegTensor<T> vregExpMax;
        MicroAPI::RegTensor<T> vregInExpSum;
        MicroAPI::RegTensor<T> vregExpSumUpdate;
        MicroAPI::RegTensor<T> vregExpEven;
        MicroAPI::RegTensor<T> vregExpOdd;
        // half / bfloat16_t
        MicroAPI::RegTensor<T2> vregExpEvenF16;
        MicroAPI::RegTensor<T2> vregExpOddF16;
        MicroAPI::RegTensor<T2> vregExpF16;
        // pse shift
        MicroAPI::RegTensor<pseShiftType> vregPseShift;
        MicroAPI::RegTensor<pseShiftType> vregPseShiftUnroll;
        MicroAPI::RegTensor<T> vregPseShiftCast;
        MicroAPI::RegTensor<T> vregPseShiftCastUnroll;

        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllB16 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllPse = MicroAPI::CreateMask<pseShiftType, MicroAPI::MaskPattern::ALL>();
        uint32_t tmpM = m;
        MicroAPI::MaskReg pregM = MicroAPI::UpdateMask<T>(tmpM);
        MicroAPI::MaskReg pregNB16 = MicroAPI::UpdateMask<T2>(pltN);
        MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(pltTailN);
        MicroAPI::MaskReg pregOriTailN = MicroAPI::UpdateMask<T>(pltOriTailN);
        MicroAPI::MaskReg pregOriTailNPse = MicroAPI::UpdateMask<pseShiftType>(pltOriTailN2);
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompareUnroll;
        MicroAPI::MaskReg pregCompareBand;
        MicroAPI::MaskReg pregCompareBandUnroll;

        // int8/fp8 quant
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant;
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuantUnroll;
        MicroAPI::RegTensor<half> vregCastB16;
        MicroAPI::RegTensor<half> vregCastB16Unroll;
        MicroAPI::RegTensor<half> vregCastRes;
        MicroAPI::RegTensor<half> vregMulsRes;
        MicroAPI::RegTensor<T> vregMulsEven;
        MicroAPI::RegTensor<T> vregMulsOdd;
        MicroAPI::RegTensor<T2> vregCast;
        MicroAPI::RegTensor<T2> vregRes;
        MicroAPI::MaskReg pregAllFp16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate<T, T>(vregMin, minValue);
        if constexpr (hasAtten == 1 && isMlaSGD) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (pregCompare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (pregCompareUnroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
        }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < rows; ++i) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant, srcUb + i * sInner);
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuantUnroll, srcUb + floatRepSize + i * sInner);

                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX, vregInputQuant, pregAll);
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputXUnroll, vregInputQuantUnroll, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputX, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputXUnroll, srcUb + floatRepSize + i * sInner);
            }

            if constexpr (IsSameType<T2, int8_t>::value ||
                IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, dequantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll, dequantScale1, pregAll);
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputX, vregInputX, scale, pregAll);  // Muls(scale)
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputXUnroll, vregInputXUnroll, scale, pregOriTailN);
            MicroAPI::Select<T>(vregInputXUnroll, vregInputXUnroll, vregMin, pregOriTailN); // 筛除actual seq length引入的脏数据

            if constexpr (hasPse == 1) {
                // fp16/bf16 -> fp32
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift, pseUb + i * psePadding);
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShiftUnroll, pseUb + floatRepSize + i * psePadding);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                    MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast, vregPseShift, pregAllPse);
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCastUnroll, vregPseShiftUnroll, pregOriTailNPse);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, vregPseShiftCast, pregAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll, vregPseShiftCastUnroll, pregOriTailN); // Avoid access out of range
                MicroAPI::Select<T>(vregInputXUnroll, vregInputXUnroll, vregMin, pregOriTailN);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSGD) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare, ((__ubuf__ uint32_t*)(maskUb + i * nPadding)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareUnroll, ((__ubuf__ uint32_t*)(maskUbUnroll + i * nPadding)));
                }
                if constexpr (isBand) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand, ((__ubuf__ uint32_t*)(maskBandUb + i * nPadding)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBandUnroll, ((__ubuf__ uint32_t*)(maskBandUbUnroll + i * nPadding)));
                    MicroAPI::Select<T>(vregSelBand, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelBandUnroll, vregMin, vregInputXUnroll, pregCompareUnroll);
                    MicroAPI::Select<T>(vregSel, vregSelBand, vregMin, pregCompareBand);
                    MicroAPI::Select<T>(vregSelUnroll, vregSelBandUnroll, vregMin, pregCompareBandUnroll);
                } else {
                    MicroAPI::Select<T>(vregSel, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelUnroll, vregMin, vregInputXUnroll, pregCompareUnroll);
                }
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregSel, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregSelUnroll, pregTailN);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregSel, vregSelUnroll, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp, pregAll);
            } else {
                MicroAPI::Select<T>(vregInputXUnrollNew,
                    vregInputXUnroll, vregMin, pregOriTailN);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregInputX, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregInputXUnrollNew, pregTailN);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregInputX, vregInputXUnrollNew, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp, pregAll);
            }
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpMaxUb, vregInputMax, uregMax, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (tmpMaxUb, uregMax, 0);

        // load history max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInMax, inMaxUb);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        // load current max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInputMax, tmpMaxUbStart);
        // max(history max, current max)
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregMax, vregInputMax, vregInMax, pregM);
        // exp_max = exp(inmax - x_max)
        MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpMax, vregInMax, vregMax, pregM);
        // store exp_max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (expMaxUb, vregExpMax, pregM);
        // store max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (maxUb, vregMax, pregM);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>
                (vregMax, maxUb + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vregInputX, vregInputXUnroll, srcUbTmp + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpEven, vregInputX, vregMax, pregAll);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpOdd, vregInputXUnroll, vregMax, pregAll);

            if constexpr (IsSameType<T2, int8_t>::value) {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<half, T, castTrait0>(vregCastB16, vregExpEven, pregAll);
                MicroAPI::Cast<half, T, castTrait1>(vregCastB16Unroll, vregExpOdd, pregAll);

                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregCastRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCastB16, (MicroAPI::RegTensor<uint16_t>&)vregCastB16Unroll,
                    pregAllFp16);
                MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes, vregCastRes, (half)quantScale1, pregAllFp16);

                MicroAPI::Cast<T2, half, castTrait0>(vregCast, vregMulsRes, pregAllFp16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vregRes, blockStride, repeatStride, pregNB16);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsEven, vregExpEven, quantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsOdd, vregExpOdd, quantScale1, pregAll);
                if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
                } else {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
                }
                MicroAPI::Or<uint8_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint8_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint8_t>&)vregExpEvenF16, (MicroAPI::RegTensor<uint8_t>&)vregExpOddF16,
                    pregAllB16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb,
                    vregRes, blockStride, repeatStride, pregNB16);
            } else {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregExpEven, pregAll);
                MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregExpOdd, pregAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>
                    ((MicroAPI::RegTensor<uint16_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpEvenF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpOddF16, pregAllB16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb, vregExpF16, blockStride,
                    repeatStride, pregNB16);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpEven, vregExpOdd, pregAll);
            MicroAPI::ReduceSum<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpSum, pregAll);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpExpSumUb, vregExpSum, uregExpSum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (tmpExpSumUb, uregExpSum, 0);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        // x_sum = sum(exp_max * in_sum + x_sum)
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInExpSum, inExpSumUb);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregExpSumBrc, tmpExpSumUbStart);
        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpSumUpdate, vregExpMax, vregInExpSum, pregM);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpSumUpdate, vregExpSumUpdate, vregExpSumBrc, pregM);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (expSumUb, vregExpSumUpdate, pregM);
    }
}

// originN = 128, Update
template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3UpdateImpl128(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    constexpr uint32_t floatRepSize = 64;
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * srcUb = (__ubuf__ MMOUTPUT_T*)inSrcTensor.GetPhyAddr();
    __ubuf__ T * srcUbTmp = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
    __ubuf__ float * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ float * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpExpSumUbStart = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + m * reduceN;
    __ubuf__ float * tmpMaxUbStart = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + m * reduceN;
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)inPseTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUbUnroll = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize].GetPhyAddr();
    __ubuf__ uint8_t * maskBandUb;
    __ubuf__ uint8_t * maskBandUbUnroll;
    if constexpr (isBand) {
        maskBandUb = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner].GetPhyAddr();
        maskBandUbUnroll = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize].GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregSel;
        MicroAPI::RegTensor<T> vregSelUnroll;
        MicroAPI::RegTensor<T> vregSelBand;
        MicroAPI::RegTensor<T> vregSelBandUnroll;
        MicroAPI::RegTensor<T> vregInputX;
        MicroAPI::RegTensor<T> vregInputXUnroll;
        MicroAPI::RegTensor<T> vregMaxTmp;
        MicroAPI::RegTensor<T> vregInputMax;
        MicroAPI::RegTensor<T> vregExpSum;
        MicroAPI::RegTensor<T> vregExpSumBrc;
        MicroAPI::RegTensor<T> vregInMax;
        MicroAPI::RegTensor<T> vregMax;
        MicroAPI::RegTensor<T> vregExpMax;
        MicroAPI::RegTensor<T> vregInExpSum;
        MicroAPI::RegTensor<T> vregExpSumUpdate;
        MicroAPI::RegTensor<T> vregExpEven;
        MicroAPI::RegTensor<T> vregExpOdd;
        // half / bfloat16_t
        MicroAPI::RegTensor<T2> vregExpEvenF16;
        MicroAPI::RegTensor<T2> vregExpOddF16;
        MicroAPI::RegTensor<T2> vregExpF16;

        // pse shift
        MicroAPI::RegTensor<pseShiftType> vregPseShift;
        MicroAPI::RegTensor<pseShiftType> vregPseShiftUnroll;
        MicroAPI::RegTensor<T> vregPseShiftCast;
        MicroAPI::RegTensor<T> vregPseShiftCastUnroll;

        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllB16 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllPse = MicroAPI::CreateMask<pseShiftType, MicroAPI::MaskPattern::ALL>();
        uint32_t tmpM = m;
        MicroAPI::MaskReg pregM = MicroAPI::UpdateMask<T>(tmpM);
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompareUnroll;
        MicroAPI::MaskReg pregCompareBand;
        MicroAPI::MaskReg pregCompareBandUnroll;

        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant;
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuantUnroll;
        MicroAPI::RegTensor<half> vregCastB16;
        MicroAPI::RegTensor<half> vregCastB16Unroll;
        MicroAPI::RegTensor<half> vregCastRes;
        MicroAPI::RegTensor<half> vregMulsRes;
        MicroAPI::RegTensor<T> vregMulsEven;
        MicroAPI::RegTensor<T> vregMulsOdd;
        MicroAPI::RegTensor<T2> vregCast;
        MicroAPI::RegTensor<T2> vregRes;
        MicroAPI::MaskReg pregAllFp16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregS8 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::VL128>();

        if constexpr (hasAtten == 1) {
            MicroAPI::Duplicate<T, T>(vregMin, minValue);
            if constexpr (isMlaSGD) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (pregCompare, ((__ubuf__ uint32_t*)(maskUb)));
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (pregCompareUnroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
            }
        }

        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < rows; ++i) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant, srcUb + i * sInner);
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuantUnroll, srcUb + floatRepSize + i * sInner);

                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX, vregInputQuant, pregAll);
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputXUnroll, vregInputQuantUnroll, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputX, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputXUnroll, srcUb + floatRepSize + i * sInner);
            }

            if constexpr (IsSameType<T2, int8_t>::value ||
                IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, dequantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll, dequantScale1, pregAll);
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputX, vregInputX, scale, pregAll);
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputXUnroll, vregInputXUnroll, scale, pregAll);

            if constexpr (hasPse == 1) {
                // fp16/bf16 -> fp32
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift, pseUb + i * sInner);
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShiftUnroll, pseUb + floatRepSize + i * sInner);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                    MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast, vregPseShift, pregAllPse);
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCastUnroll, vregPseShiftUnroll, pregAllPse);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, vregPseShiftCast, pregAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputXUnroll, vregInputXUnroll, vregPseShiftCastUnroll, pregAll); // Avoid access out of range
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSGD) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare, ((__ubuf__ uint32_t*)(maskUb + i * sInner)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareUnroll, ((__ubuf__ uint32_t*)(maskUbUnroll + i * sInner)));
                }
                if constexpr (isBand) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand, ((__ubuf__ uint32_t*)(maskBandUb + i * sInner)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBandUnroll, ((__ubuf__ uint32_t*)(maskBandUbUnroll + i * sInner)));
                    MicroAPI::Select<T>(vregSelBand, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelBandUnroll, vregMin, vregInputXUnroll, pregCompareUnroll);
                    MicroAPI::Select<T>(vregSel, vregSelBand, vregMin, pregCompareBand);
                    MicroAPI::Select<T>(vregSelUnroll, vregSelBandUnroll, vregMin, pregCompareBandUnroll);
                } else {
                    MicroAPI::Select<T>(vregSel, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelUnroll, vregMin, vregInputXUnroll, pregCompareUnroll);
                }
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregSel, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregSelUnroll, pregAll);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregSel, vregSelUnroll, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregInputX, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregInputXUnroll, pregAll);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregInputX, vregInputXUnroll, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp, pregAll);
            }
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpMaxUb, vregInputMax, uregMax, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (tmpMaxUb, uregMax, 0);
        // load history max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInMax, inMaxUb);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        // load current max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInputMax, tmpMaxUbStart);
        // max(history max, current max)
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregMax, vregInputMax, vregInMax, pregM);
        // exp_max = exp(inmax - x_max)
        MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpMax, vregInMax, vregMax, pregM);
        // store exp_max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (expMaxUb, vregExpMax, pregM);
        // store max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (maxUb, vregMax, pregM);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>
                (vregMax, maxUb + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vregInputX, vregInputXUnroll, srcUbTmp + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpEven, vregInputX, vregMax, pregAll);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpOdd, vregInputXUnroll, vregMax, pregAll);

            if constexpr (IsSameType<T2, int8_t>::value) {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<half, T, castTrait0>(vregCastB16, vregExpEven, pregAll);
                MicroAPI::Cast<half, T, castTrait1>(vregCastB16Unroll, vregExpOdd, pregAll);

                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregCastRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCastB16, (MicroAPI::RegTensor<uint16_t>&)vregCastB16Unroll,
                    pregAllFp16);
                MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes, vregCastRes, (half)quantScale1, pregAllB16);

                MicroAPI::Cast<T2, half, castTrait0>(vregCast, vregMulsRes, pregAllFp16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vregRes, blockStride, repeatStride, pregS8);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsEven, vregExpEven, quantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsOdd, vregExpOdd, quantScale1, pregAll);
                if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
                } else {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
                }
                MicroAPI::Or<uint8_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint8_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint8_t>&)vregExpEvenF16, (MicroAPI::RegTensor<uint8_t>&)vregExpOddF16,
                    pregAllB16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb,
                    vregRes, blockStride, repeatStride, pregS8);
            } else {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregExpEven, pregAll);
                MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregExpOdd, pregAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>
                    ((MicroAPI::RegTensor<uint16_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpEvenF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpOddF16, pregAllB16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb, vregExpF16, blockStride,
                    repeatStride, pregAllB16);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpEven, vregExpOdd, pregAll);
            MicroAPI::ReduceSum<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpSum, pregAll);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpExpSumUb, vregExpSum, uregExpSum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (tmpExpSumUb, uregExpSum, 0);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        // x_sum = sum(exp_max * in_sum + x_sum)
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInExpSum, inExpSumUb);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregExpSumBrc, tmpExpSumUbStart);
        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpSumUpdate, vregExpMax, vregInExpSum, pregM);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpSumUpdate, vregExpSumUpdate, vregExpSumBrc, pregM);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (expSumUb, vregExpSumUpdate, pregM);
    }
}

// originN <= 64, Update
template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3UpdateImpl64(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    constexpr uint32_t floatRepSize = 64;
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint32_t blockU8 = 32;
    constexpr uint32_t blockB16 = 16;
    const uint32_t nPadding = (originN + blockU8 - 1) / blockU8 * blockU8;
    const uint32_t psePadding = (originN + blockB16 - 1) / blockB16 * blockB16;
    constexpr uint16_t repeatStride = 1;
    uint32_t pltOriginalN = originN;
    uint32_t pltOriginalN2 = originN * (sizeof(T) / sizeof(pseShiftType));
    uint32_t pltSrcN = sInner;
    uint32_t pltSrcN16 = sInner;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * srcUb = (__ubuf__ MMOUTPUT_T*)inSrcTensor.GetPhyAddr();
    __ubuf__ T * srcUbTmp = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
    __ubuf__ float * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ float * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpExpSumUbStart = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + m * reduceN;
    __ubuf__ float * tmpMaxUbStart = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + m * reduceN;
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)inPseTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskBandUb;
    if constexpr (isBand) {
        maskBandUb = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner].GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregSel;
        MicroAPI::RegTensor<T> vregSelBand;
        MicroAPI::RegTensor<T> vregInputX;
        MicroAPI::RegTensor<T> vregInputXUnroll;
        MicroAPI::RegTensor<T> vregInputMax;
        MicroAPI::RegTensor<T> vregExp;
        MicroAPI::RegTensor<T> vregExpSum;
        MicroAPI::RegTensor<T> vregExpSumBrc;
        MicroAPI::RegTensor<T> vregInMax;
        MicroAPI::RegTensor<T> vregMax;
        MicroAPI::RegTensor<T> vregExpMax;
        MicroAPI::RegTensor<T> vregInExpSum;
        MicroAPI::RegTensor<T> vregExpSumUpdate;

        // half / bfloat16_t
        MicroAPI::RegTensor<T2> vregExpEvenF16;
        MicroAPI::RegTensor<T2> vregExpF16;
        MicroAPI::RegTensor<T2> vregDstEvenF16;
        MicroAPI::RegTensor<T2> vregDstOddF16;
        MicroAPI::RegTensor<T2> vregDummyF16;

        // pse shift
        MicroAPI::RegTensor<pseShiftType> vregPseShift;
        MicroAPI::RegTensor<T> vregPseShiftCast;

        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllB16 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
        uint32_t tmpM = m;
        MicroAPI::MaskReg pregM = MicroAPI::UpdateMask<T>(tmpM);
        MicroAPI::MaskReg pregSrcN = MicroAPI::UpdateMask<T>(pltSrcN);
        MicroAPI::MaskReg pregSrcNB16 = MicroAPI::UpdateMask<T2>(pltSrcN16);
        MicroAPI::MaskReg pregOriSrcN = MicroAPI::UpdateMask<T>(pltOriginalN);
        MicroAPI::MaskReg pregOriSrcNPse = MicroAPI::UpdateMask<pseShiftType>(pltOriginalN2);
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompareBand;

        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant;
        MicroAPI::RegTensor<half> vregCastB16;
        MicroAPI::RegTensor<half> vregCastRes;
        MicroAPI::RegTensor<half> vregMulsRes;
        MicroAPI::RegTensor<T> vregMulsB32;
        MicroAPI::RegTensor<T2> vregCast;
        MicroAPI::RegTensor<T2> vregRes;
        MicroAPI::MaskReg pregAllFp16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();

        if constexpr (hasAtten == 1) {
            MicroAPI::Duplicate<T, T>(vregMin, minValue);
            if constexpr (isMlaSGD) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (pregCompare, ((__ubuf__ uint32_t*)(maskUb)));
            }
        }

        // x_max = max(src, axis=-1, keepdims=True)
        for (uint16_t i = 0; i < rows; ++i) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant, srcUb + i * sInner);

                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX, vregInputQuant, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputX, srcUb + i * sInner);
            }

            if constexpr (IsSameType<T2, int8_t>::value ||
                IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, dequantScale1, pregAll);
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregInputX, vregInputX, scale, pregOriSrcN);  // Muls(scale)
            MicroAPI::Select<T>(vregInputX, vregInputX, vregMin, pregOriSrcN); // 筛除actual seq length引入的脏数据

            if constexpr (hasPse == 1) {
                // fp16/bf16 -> fp32
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift, pseUb + i * psePadding);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                    MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast, vregPseShift, pregOriSrcNPse);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, vregPseShiftCast, pregOriSrcN);
                MicroAPI::Select<T>(vregInputX, vregInputX, vregMin, pregOriSrcN);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSGD) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare, ((__ubuf__ uint32_t*)(maskUb + i * nPadding)));
                }
                if constexpr (isBand) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand, ((__ubuf__ uint32_t*)(maskBandUb + i * nPadding)));
                    MicroAPI::Select<T>(vregSelBand, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSel, vregSelBand, vregMin, pregCompareBand);
                } else {
                    MicroAPI::Select<T>(vregSel, vregMin, vregInputX, pregCompare);
                }
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregSel, pregSrcN);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregSel, pregOriSrcN);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregInputX, pregSrcN);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregInputX, pregOriSrcN);
            }
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpMaxUb, vregInputMax, uregMax, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (tmpMaxUb, uregMax, 0);

        // load history max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInMax, inMaxUb);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        // load current max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInputMax, tmpMaxUbStart);
        // max(history max, current max)
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregMax, vregInputMax, vregInMax, pregM);
        // exp_max = exp(inmax - x_max)
        MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpMax, vregInMax, vregMax, pregM);
        // store exp_max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (expMaxUb, vregExpMax, pregM);
        // store max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (maxUb, vregMax, pregM);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>
                (vregMax, maxUb + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                (vregInputX, srcUbTmp + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExp, vregInputX, vregMax, pregOriSrcN);

            if constexpr (IsSameType<T2, int8_t>::value) {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<half, T, castTrait0>(vregCastB16, vregExp, pregAll);

                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)vregCastRes,
                    (MicroAPI::RegTensor<uint32_t>&)vregCastB16);
                MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes, vregCastRes, (half)quantScale1, pregAllFp16);

                MicroAPI::Cast<T2, half, castTrait0>(vregCast, vregMulsRes, pregAllFp16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vregRes, blockStride, repeatStride, pregSrcNB16);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsB32, vregExp, quantScale1, pregAll);
                if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsB32, pregAll);
                } else {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsB32, pregAll);
                }
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint32_t>&)vregExpEvenF16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb,
                    vregRes, blockStride, repeatStride, pregSrcNB16);
            } else {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T2, T, castTrait0>(vregExpF16, vregExp, pregAllB16);
                MicroAPI::DeInterleave<uint16_t>((MicroAPI::RegTensor<uint16_t> &)vregDstEvenF16,
                    (MicroAPI::RegTensor<uint16_t> &)vregDstOddF16,
                    (MicroAPI::RegTensor<uint16_t> &)vregExpF16,
                    (MicroAPI::RegTensor<uint16_t> &)vregDummyF16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb, vregDstEvenF16, blockStride,
                    repeatStride, pregSrcNB16);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::ReduceSum<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExp, pregOriSrcN);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpExpSumUb, vregExpSum, uregExpSum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (tmpExpSumUb, uregExpSum, 0);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        // x_sum = sum(exp_max * in_sum + x_sum)
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInExpSum, inExpSumUb);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregExpSumBrc, tmpExpSumUbStart);
        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpSumUpdate, vregExpMax, vregInExpSum, pregM);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpSumUpdate, vregExpSumUpdate, vregExpSumBrc, pregM);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (expSumUb, vregExpSumUpdate, pregM);
    }
}

// 128 < originN <= 256, Update
template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3UpdateGeneralImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    constexpr uint32_t floatRepSize = 64;
	constexpr uint32_t floatRepSize2 = 128;
    constexpr uint32_t floatRepSize3 = 192;
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint32_t blockU8 = 32;
    constexpr uint32_t blockB16 = 16;
    const uint32_t nPadding = (originN + blockU8 - 1) / blockU8 * blockU8;
    const uint32_t psePadding = (originN + blockB16 - 1) / blockB16 * blockB16;
    constexpr uint16_t repeatStride = 1;

    const uint32_t oriTailN1 = (originN - floatRepSize2 < floatRepSize) ? (originN - floatRepSize2) : floatRepSize;
	const uint32_t oriTailN2 = (originN <= floatRepSize3) ? 0 : (originN - floatRepSize3);
    const uint32_t tailN1 = sInner - floatRepSize2;
	const uint32_t tailN2 = sInner - floatRepSize3;
    uint32_t pltOriTailN1 = oriTailN1;
	uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltOriTailPse1 = oriTailN1 * (sizeof(T) / sizeof(pseShiftType));
	uint32_t pltOriTailPse2 = oriTailN2 * (sizeof(T) / sizeof(pseShiftType));
    uint32_t pltTailN1 = tailN1;
	uint32_t pltTailN2 = tailN2;
    uint32_t pltN = sInner;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
	__ubuf__ T2 * expUb1 = ((__ubuf__ T2*)dstTensor.GetPhyAddr()) + (vsOuter + 1) * (sInner >> 1U);
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T * srcUb = (__ubuf__ MMOUTPUT_T*)inSrcTensor.GetPhyAddr();
    __ubuf__ T * srcUbTmp = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
    __ubuf__ float * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ float * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpExpSumUbStart = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + m * reduceN;
    __ubuf__ float * tmpMaxUbStart = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + m * reduceN;
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)inPseTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    __ubuf__ uint8_t * maskUb1 = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize].GetPhyAddr();
	__ubuf__ uint8_t * maskUb2 = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize2].GetPhyAddr();
	__ubuf__ uint8_t * maskUb3 = (__ubuf__ uint8_t*)inMaskTensor[floatRepSize3].GetPhyAddr();
    __ubuf__ uint8_t * maskBandUb;
    __ubuf__ uint8_t * maskBandUb1;
	__ubuf__ uint8_t * maskBandUb2;
	__ubuf__ uint8_t * maskBandUb3;
    if constexpr (isBand) {
        maskBandUb = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner].GetPhyAddr();
        maskBandUb1 = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize].GetPhyAddr();
		maskBandUb2 = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize2].GetPhyAddr();
		maskBandUb3 = (__ubuf__ uint8_t*)inMaskTensor[vsOuter * sInner + floatRepSize3].GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregSel;
        MicroAPI::RegTensor<T> vregSel1;
		MicroAPI::RegTensor<T> vregSel2;
        MicroAPI::RegTensor<T> vregSel3;
        MicroAPI::RegTensor<T> vregSelBand;
        MicroAPI::RegTensor<T> vregSelBand1;
		MicroAPI::RegTensor<T> vregSelBand2;
		MicroAPI::RegTensor<T> vregSelBand3;
        MicroAPI::RegTensor<T> vregInputX;
        MicroAPI::RegTensor<T> vregInputX1;
		MicroAPI::RegTensor<T> vregInputX2;
		MicroAPI::RegTensor<T> vregInputX3;
        MicroAPI::RegTensor<T> vregInputX2New;
		MicroAPI::RegTensor<T> vregInputX3New;
        MicroAPI::RegTensor<T> vregMaxTmp;
		MicroAPI::RegTensor<T> vregMaxTmp1;
		MicroAPI::RegTensor<T> vregMaxTmp2;
        MicroAPI::RegTensor<T> vregInputMax;
        MicroAPI::RegTensor<T> vregMaxBrc;
        MicroAPI::RegTensor<T> vregExpSum;
		MicroAPI::RegTensor<T> vregExpSum1;
        MicroAPI::RegTensor<T> vregExpSumBrc;
        MicroAPI::RegTensor<T> vregInMax;
        MicroAPI::RegTensor<T> vregMax;
        MicroAPI::RegTensor<T> vregExpMax;
        MicroAPI::RegTensor<T> vregInExpSum;
        MicroAPI::RegTensor<T> vregExpSumUpdate;
        MicroAPI::RegTensor<T> vregExpEven;
		MicroAPI::RegTensor<T> vregExpEven1;
        MicroAPI::RegTensor<T> vregExpOdd;
		MicroAPI::RegTensor<T> vregExpOdd1;
        // half / bfloat16_t
        MicroAPI::RegTensor<T2> vregExpEvenF16;
		MicroAPI::RegTensor<T2> vregExpEvenF161;
        MicroAPI::RegTensor<T2> vregExpOddF16;
		MicroAPI::RegTensor<T2> vregExpOddF161;
        MicroAPI::RegTensor<T2> vregExpF16;
		MicroAPI::RegTensor<T2> vregExpF161;
        // pse shift
        MicroAPI::RegTensor<pseShiftType> vregPseShift;
        MicroAPI::RegTensor<pseShiftType> vregPseShift1;
		MicroAPI::RegTensor<pseShiftType> vregPseShift2;
		MicroAPI::RegTensor<pseShiftType> vregPseShift3;
        MicroAPI::RegTensor<T> vregPseShiftCast;
        MicroAPI::RegTensor<T> vregPseShiftCast1;
		MicroAPI::RegTensor<T> vregPseShiftCast2;
		MicroAPI::RegTensor<T> vregPseShiftCast3;

        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllB16 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAllPse = MicroAPI::CreateMask<pseShiftType, MicroAPI::MaskPattern::ALL>();
        uint32_t tmpM = m;
        MicroAPI::MaskReg pregM = MicroAPI::UpdateMask<T>(tmpM);
        MicroAPI::MaskReg pregNB16 = MicroAPI::UpdateMask<T2>(pltN);
        MicroAPI::MaskReg pregTailN1 = MicroAPI::UpdateMask<T>(pltTailN1);
		MicroAPI::MaskReg pregTailN2 = MicroAPI::UpdateMask<T>(pltTailN2);
        MicroAPI::MaskReg pregOriTailN1 = MicroAPI::UpdateMask<T>(pltOriTailN1);
		MicroAPI::MaskReg pregOriTailN2 = MicroAPI::UpdateMask<T>(pltOriTailN2);
        MicroAPI::MaskReg pregOriTailNPse1 = MicroAPI::UpdateMask<pseShiftType>(pltOriTailPse1);
		MicroAPI::MaskReg pregOriTailNPse2 = MicroAPI::UpdateMask<pseShiftType>(pltOriTailPse2);
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompare1;
		MicroAPI::MaskReg pregCompare2;
		MicroAPI::MaskReg pregCompare3;
        MicroAPI::MaskReg pregCompareBand;
        MicroAPI::MaskReg pregCompareBand1;
		MicroAPI::MaskReg pregCompareBand2;
		MicroAPI::MaskReg pregCompareBand3;

        // int8/fp8 quant
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant;
        MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant1;
		MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant2;
		MicroAPI::RegTensor<MMOUTPUT_T> vregInputQuant3;
        MicroAPI::RegTensor<half> vregCastB16;
        MicroAPI::RegTensor<half> vregCastB161;
		MicroAPI::RegTensor<half> vregCastB162;
		MicroAPI::RegTensor<half> vregCastB163;
        MicroAPI::RegTensor<half> vregCastRes;
		MicroAPI::RegTensor<half> vregCastRes1;
        MicroAPI::RegTensor<half> vregMulsRes;
		MicroAPI::RegTensor<half> vregMulsRes1;
        MicroAPI::RegTensor<T> vregMulsEven;
		MicroAPI::RegTensor<T> vregMulsEven1;
        MicroAPI::RegTensor<T> vregMulsOdd;
		MicroAPI::RegTensor<T> vregMulsOdd1;
        MicroAPI::RegTensor<T2> vregCast;
		MicroAPI::RegTensor<T2> vregCast1;
        MicroAPI::RegTensor<T2> vregRes;
		MicroAPI::RegTensor<T2> vregRes1;
        MicroAPI::MaskReg pregAllFp16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate<T, T>(vregMin, minValue);
        if constexpr (hasAtten == 1 && isMlaSGD) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>(pregCompare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>(pregCompare1, ((__ubuf__ uint32_t*)(maskUb1)));
			MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>(pregCompare2, ((__ubuf__ uint32_t*)(maskUb2)));
			MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>(pregCompare3, ((__ubuf__ uint32_t*)(maskUb3)));
        }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < rows; ++i) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant, srcUb + i * sInner);
                MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant1, srcUb + floatRepSize + i * sInner);
				MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant2, srcUb + floatRepSize2 + i * sInner);
				MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>
                    (vregInputQuant3, srcUb + floatRepSize3 + i * sInner);

                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX, vregInputQuant, pregAll);
                MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX1, vregInputQuant1, pregAll);
				MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX2, vregInputQuant2, pregAll);
				MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vregInputX3, vregInputQuant3, pregAll);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInputX, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInputX1, srcUb + floatRepSize + i * sInner);
				MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInputX2, srcUb + floatRepSize2 + i * sInner);
				MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInputX3, srcUb + floatRepSize3 + i * sInner);
            }

            if constexpr (IsSameType<T2, int8_t>::value ||
                IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, dequantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX1, vregInputX1, dequantScale1, pregAll);
				MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX2, vregInputX2, dequantScale1, pregAll);
				MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX3, vregInputX3, dequantScale1, pregAll);
            }

            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, scale, pregAll);  // Muls(scale)
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX1, vregInputX1, scale, pregAll);
			MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX2, vregInputX2, scale, pregOriTailN1);
			MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX3, vregInputX3, scale, pregOriTailN2);
            MicroAPI::Select<T>(vregInputX2, vregInputX2, vregMin, pregOriTailN1); // 筛除actual seq length引入的脏数据
			MicroAPI::Select<T>(vregInputX3, vregInputX3, vregMin, pregOriTailN2); // 筛除actual seq length引入的脏数据

            if constexpr (hasPse == 1) {
                // fp16/bf16 -> fp32
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift, pseUb + i * psePadding);
                MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift1, pseUb + floatRepSize + i * psePadding);
				MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift2, pseUb + floatRepSize2 + i * psePadding);
				MicroAPI::DataCopy<pseShiftType, MicroAPI::LoadDist::DIST_UNPACK_B16>
                    (vregPseShift3, pseUb + floatRepSize3 + i * psePadding);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                    MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast, vregPseShift, pregAllPse);
                MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast1, vregPseShift1, pregAllPse);
				MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast2, vregPseShift2, pregOriTailNPse1);
				MicroAPI::Cast<T, pseShiftType, castTrait>(vregPseShiftCast3, vregPseShift3, pregOriTailNPse2);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX, vregInputX, vregPseShiftCast, pregAll);
				MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX1, vregInputX1, vregPseShiftCast1, pregAll);
				MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX2, vregInputX2, vregPseShiftCast2, pregOriTailN1);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputX3, vregInputX3, vregPseShiftCast3, pregOriTailN2); // Avoid access out of range
                MicroAPI::Select<T>(vregInputX2, vregInputX2, vregMin, pregOriTailN1);
				MicroAPI::Select<T>(vregInputX3, vregInputX3, vregMin, pregOriTailN2);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSGD) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare, ((__ubuf__ uint32_t*)(maskUb + i * nPadding)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare1, ((__ubuf__ uint32_t*)(maskUb1 + i * nPadding)));
					MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare2, ((__ubuf__ uint32_t*)(maskUb2 + i * nPadding)));
					MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompare3, ((__ubuf__ uint32_t*)(maskUb3 + i * nPadding)));
                }
                if constexpr (isBand) {
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand, ((__ubuf__ uint32_t*)(maskBandUb + i * nPadding)));
                    MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand1, ((__ubuf__ uint32_t*)(maskBandUb1 + i * nPadding)));
					MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand2, ((__ubuf__ uint32_t*)(maskBandUb2 + i * nPadding)));
					MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                        (pregCompareBand3, ((__ubuf__ uint32_t*)(maskBandUb3 + i * nPadding)));
                    MicroAPI::Select<T>(vregSelBand, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSelBand1, vregMin, vregInputX1, pregCompare1);
					MicroAPI::Select<T>(vregSelBand2, vregMin, vregInputX2, pregCompare2);
					MicroAPI::Select<T>(vregSelBand3, vregMin, vregInputX3, pregCompare3);
                    MicroAPI::Select<T>(vregSel, vregSelBand, vregMin, pregCompareBand);
                    MicroAPI::Select<T>(vregSel1, vregSelBand1, vregMin, pregCompareBand1);
					MicroAPI::Select<T>(vregSel2, vregSelBand2, vregMin, pregCompareBand2);
					MicroAPI::Select<T>(vregSel3, vregSelBand3, vregMin, pregCompareBand3);
                } else {
                    MicroAPI::Select<T>(vregSel, vregMin, vregInputX, pregCompare);
                    MicroAPI::Select<T>(vregSel1, vregMin, vregInputX1, pregCompare1);
					MicroAPI::Select<T>(vregSel2, vregMin, vregInputX2, pregCompare2);
					MicroAPI::Select<T>(vregSel3, vregMin, vregInputX3, pregCompare3);
                }
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregSel, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregSel1, pregAll);
				MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize2 + i * sInner, vregSel2, pregTailN1);
				MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize3 + i * sInner, vregSel3, pregTailN2);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregSel, vregSel1, pregAll);
				MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp1, vregSel2, vregSel3, pregAll);
				MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp2, vregMaxTmp, vregMaxTmp1, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp2, pregAll);
            } else {
                MicroAPI::Select<T>(vregInputX2New, vregInputX2, vregMin, pregOriTailN1);
				MicroAPI::Select<T>(vregInputX3New, vregInputX3, vregMin, pregOriTailN2);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + i * sInner, vregInputX, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize + i * sInner, vregInputX1, pregAll);
				MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize2 + i * sInner, vregInputX2New, pregTailN1);
				MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUbTmp + floatRepSize3 + i * sInner, vregInputX3New, pregTailN2);
                MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp, vregInputX, vregInputX1, pregAll);
				MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp1, vregInputX2New, vregInputX3New, pregAll);
				MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregMaxTmp2, vregMaxTmp, vregMaxTmp1, pregAll);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vregInputMax, vregMaxTmp2, pregAll);
            }
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpMaxUb, vregInputMax, uregMax, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (tmpMaxUb, uregMax, 0);

        // load history max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInMax, inMaxUb);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        // load current max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInputMax, tmpMaxUbStart);
        // max(history max, current max)
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregMax, vregInputMax, vregInMax, pregM);
        // exp_max = exp(inmax - x_max)
        MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpMax, vregInMax, vregMax, pregM);
        // store exp_max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (expMaxUb, vregExpMax, pregM);
        // store max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (maxUb, vregMax, pregM);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>
                (vregMax, maxUb + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vregInputX, vregInputX1, srcUbTmp + i * sInner);
			MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vregInputX2, vregInputX3, srcUbTmp + floatRepSize2 + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpEven, vregInputX, vregMax, pregAll);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpOdd, vregInputX1, vregMax, pregAll);
			MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpEven1, vregInputX2, vregMax, pregAll);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                MicroAPI::MaskMergeMode::ZEROING>(vregExpOdd1, vregInputX3, vregMax, pregAll);

            if constexpr (IsSameType<T2, int8_t>::value) {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<half, T, castTrait0>(vregCastB16, vregExpEven, pregAll);
                MicroAPI::Cast<half, T, castTrait1>(vregCastB161, vregExpOdd, pregAll);
				MicroAPI::Cast<half, T, castTrait0>(vregCastB162, vregExpEven1, pregAll);
                MicroAPI::Cast<half, T, castTrait1>(vregCastB163, vregExpOdd1, pregAll);

                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregCastRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCastB16, (MicroAPI::RegTensor<uint16_t>&)vregCastB161,
                    pregAllFp16);
				MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregCastRes1,
                    (MicroAPI::RegTensor<uint16_t>&)vregCastB162, (MicroAPI::RegTensor<uint16_t>&)vregCastB163,
                    pregAllFp16);
                MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes, vregCastRes, (half)quantScale1, pregAllFp16);
				MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregMulsRes1, vregCastRes1, (half)quantScale1, pregAllFp16);

                MicroAPI::Cast<T2, half, castTrait0>(vregCast, vregMulsRes, pregAllFp16);
				MicroAPI::Cast<T2, half, castTrait1>(vregCast1, vregMulsRes1, pregAllFp16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast);
				MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes1,
                    (MicroAPI::RegTensor<uint16_t>&)vregCast1);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vregRes, blockStride, repeatStride, pregNB16);
				MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb1),
                    vregRes1, blockStride, repeatStride, pregNB16);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsEven, vregExpEven, quantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsOdd, vregExpOdd, quantScale1, pregAll);
				MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsEven1, vregExpEven1, quantScale1, pregAll);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregMulsOdd1, vregExpOdd1, quantScale1, pregAll);
                if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
					MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF161, vregMulsEven1, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF161, vregMulsOdd1, pregAll);
                } else {
                    static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
                    MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregMulsEven, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregMulsOdd, pregAll);
					MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF161, vregMulsEven1, pregAll);
                    MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF161, vregMulsOdd1, pregAll);
                }
                MicroAPI::Or<uint8_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint8_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint8_t>&)vregExpEvenF16, (MicroAPI::RegTensor<uint8_t>&)vregExpOddF16,
                    pregAllB16);
				MicroAPI::Or<uint8_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint8_t>&)vregExpF161,
                    (MicroAPI::RegTensor<uint8_t>&)vregExpEvenF161, (MicroAPI::RegTensor<uint8_t>&)vregExpOddF161,
                    pregAllB16);
                MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF16);
				MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vregRes1,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpF161);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb,
                    vregRes, blockStride, repeatStride, pregNB16);
				MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb1,
                    vregRes1, blockStride, repeatStride, pregNB16);
            } else {
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF16, vregExpEven, pregAll);
                MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF16, vregExpOdd, pregAll);
				MicroAPI::Cast<T2, T, castTrait0>(vregExpEvenF161, vregExpEven1, pregAll);
                MicroAPI::Cast<T2, T, castTrait1>(vregExpOddF161, vregExpOdd1, pregAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>
                    ((MicroAPI::RegTensor<uint16_t>&)vregExpF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpEvenF16,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpOddF16, pregAllB16);
				MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>
                    ((MicroAPI::RegTensor<uint16_t>&)vregExpF161,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpEvenF161,
                    (MicroAPI::RegTensor<uint16_t>&)vregExpOddF161, pregAllB16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb, vregExpF16, blockStride,
                    repeatStride, pregNB16);
				MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(expUb1, vregExpF161, blockStride,
                    repeatStride, pregNB16);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpEven, vregExpOdd, pregAll);
			MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum1, vregExpEven1, vregExpOdd1, pregAll);
			MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpSum, vregExpSum1, pregAll);
            MicroAPI::ReduceSum<T, T, MicroAPI::MaskMergeMode::ZEROING>
                (vregExpSum, vregExpSum, pregAll);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpExpSumUb, vregExpSum, uregExpSum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (tmpExpSumUb, uregExpSum, 0);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        // x_sum = sum(exp_max * in_sum + x_sum)
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregInExpSum, inExpSumUb);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
            (vregExpSumBrc, tmpExpSumUbStart);
        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpSumUpdate, vregExpMax, vregInExpSum, pregM);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>
            (vregExpSumUpdate, vregExpSumUpdate, vregExpSumBrc, pregM);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
            (expSumUb, vregExpSumUpdate, pregM);
    }
}

template <typename T, typename T2, typename pseShiftType, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false, bool isBand = false,
    bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3Update8(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float dequantScale1, float quantScale1)
{
    // mode 1: originN = 128
    if constexpr (mode == 1) {
        SoftmaxFlashV3UpdateImpl128<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    } else if constexpr (mode == 2) { // mode 2: 0 < originN <= 64
        SoftmaxFlashV3UpdateImpl64<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    } else if constexpr (mode == 0) { // mode 0: 64 < originN < 128
        SoftmaxFlashV3UpdateGeneralImpl128<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    } else { // mode 3: 128 < originN <= 256
        SoftmaxFlashV3UpdateGeneralImpl256<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    }
}

/*
 * @ingroup SoftmaxFlashV3
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] inExpSumTensor, in sum(exp(x-max)) of last softmax result
 * @param [in] inMaxTensor, in max value of last softmax result
 * @param [in] maskTensor, atten mask LocalTensor, each line padding to 32, padding value is 1
 * @param [in] pseTensor, reserved
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] originN, input origin colums, support range for sInner is: 0 < sInner <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isBmm2Concat, reserved
 * @param [in] isUpdate, enable flash mode
 * @param [in] mode
 *  mode 0: 64 < originN <= 128, and originN is not 8 aligned
 *  mode 1: 64 < originN <= 128, and originN is 8 aligned
 *  mode 2: 0 < originN <= 64
 * @param [in] hasAtten, indicates whether there is atten_mask
 * @param [in] hasPse, indicates whether there is pse_shift
 */
template <typename T, typename T2, typename pseShiftType, bool isUpdate = false, uint8_t mode = 0, bool hasAtten = false, bool hasPse = false,
    bool isBand = false, bool isMlaSGD = false, bool isBmm2Concat = false, uint32_t vsOuter = 0, uint32_t sInner = 0, typename MMOUTPUT_T>
__aicore__ inline void SoftmaxFlashV3_VF(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<MMOUTPUT_T>& inSrcTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<pseShiftType>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, float dequantScale1, float quantScale1)
{
    constexpr uint32_t blockU8 = 32;
    uint32_t blockN = 0;
    if constexpr (IsSameType<T2, int8_t>::value ||
        IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
        blockN = 32;
    } else {
        blockN = 16;
    }
    uint16_t blockStride = 0;
    if constexpr (isBmm2Concat) {
        blockStride = vsOuter * blockN * sizeof(T2) / blockU8 + 1;
    } else {
        blockStride = m * blockN * sizeof(T2) / blockU8;
    }
    if constexpr (!isUpdate) {
        SoftmaxFlashV3NoUpdate8<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    } else {
        SoftmaxFlashV3Update8<T, T2, pseShiftType, mode, hasAtten, hasPse, isBand, isMlaSGD, isBmm2Concat, vsOuter, sInner, MMOUTPUT_T>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, dequantScale1, quantScale1);
    }
}

} // namespace

#endif // VF_SOFTMAX_FLASH_V3_H
