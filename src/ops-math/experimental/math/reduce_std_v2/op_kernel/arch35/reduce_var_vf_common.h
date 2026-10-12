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
 * \file reduce_var_vf_common.h
 * \brief reduce var vf common
 */

#ifndef _REDUCE_VAR_VF_COMMON_H_
#define _REDUCE_VAR_VF_COMMON_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace ReduceOpTmpl
{
using namespace AscendC;

using AscendC::MicroAPI::Cast;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::DataCopy;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using AscendC::MicroAPI::MaskPattern;

constexpr uint32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);
constexpr int32_t DICHOTOMY_ADD_COEFF = 2;
constexpr uint32_t ROW_TWO_OFFSET = 2;
constexpr uint32_t ROW_THREE_OFFSET = 3;
constexpr uint32_t ROW_FOUR_OFFSET = 4;
constexpr uint32_t SCALE_COEF_FOUR = 4;
constexpr uint64_t CONST_SIXTY_FOUR = 64;

__aicore__ inline uint64_t FindNextPower2(const uint64_t value)
{
    if (value <= Ops::Base::ReduceOpTmpl::CONST1) {
        return Ops::Base::ReduceOpTmpl::CONST1;
    } else {
        const uint64_t num = value - Ops::Base::ReduceOpTmpl::CONST1;
        const uint64_t pow = CONST_SIXTY_FOUR - AscendC::ScalarCountLeadingZero(num);
        return (Ops::Base::ReduceOpTmpl::CONST1 << pow);
    }
}

constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {AscendC::MicroAPI::RegLayout::ZERO,
                                                                  AscendC::MicroAPI::SatMode::UNKNOWN,
                                                                  MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__local_mem__ T* input, RegTensor<float>& dst, MaskReg& preg,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(__local_mem__ T *src1, __local_mem__ T *src2, RegTensor<float> &dst1,
        RegTensor<float> &dst2, MaskReg &dst1Preg, MaskReg &dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half *)(src1) + (src1Offset)));
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half *)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16Q;
        RegTensor<bfloat16_t> xFp16R;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q,
            ((__local_mem__ bfloat16_t *)(src1) + (src1Offset)));
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R,
            ((__local_mem__ bfloat16_t *)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        DataCopy(dst1, ((__local_mem__ float *)(src1) + (src1Offset)));
        DataCopy(dst2, ((__local_mem__ float *)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(__local_mem__ T* output, RegTensor<float>& src, MaskReg& preg,
                                               uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> oFp16;
        Cast<half, float, castTraitB322B16>(oFp16, src, preg);
        DataCopy<half, StoreDist::DIST_PACK_B32>((__local_mem__ half*)(output) + offset, oFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> oBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(oBf16, src, preg);
        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>((__local_mem__ bfloat16_t*)(output) + offset, oBf16, preg);
    } else {
        DataCopy((__local_mem__ float*)(output) + offset, src, preg);
    }
}

__aicore__ inline void DichotomyAdd(RegTensor<float>& dstReg, __local_mem__ float* src, uint16_t outerLoop,
                                    uint16_t innerLoop, uint32_t lastNum)
{
    RegTensor<float> tmpReg1;
    RegTensor<float> tmpReg2;
    RegTensor<float> tmpReg3;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t k = 0; k < outerLoop; k++) {
        innerLoop = innerLoop / DICHOTOMY_ADD_COEFF;
        for (uint16_t i = 0; i < innerLoop; i++) {
            DataCopy(tmpReg1, src + i * VL_FP32);
            DataCopy(tmpReg2, src + (i + innerLoop) * VL_FP32);
            Add(tmpReg3, tmpReg1, tmpReg2, pregMain);
            DataCopy(src + i * VL_FP32, tmpReg3, pregMain);
        }
        AscendC::MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    uint32_t sreg0 = lastNum;
    MaskReg pregLoop = UpdateMask<float>(sreg0);
    DataCopy(tmpReg3, src);
    ReduceSum(dstReg, tmpReg3, pregLoop);
}

template <typename T>
__aicore__ inline void StoreOneElementForDtypeT(__local_mem__ T* output, RegTensor<float>& src, MaskReg& preg,
                                                uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> oFp16;
        Cast<half, float, castTraitB322B16>(oFp16, src, preg);
        DataCopy<half, StoreDist::DIST_FIRST_ELEMENT_B16>((__local_mem__ half*)(output) + offset, oFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> oBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(oBf16, src, preg);
        DataCopy<bfloat16_t, StoreDist::DIST_FIRST_ELEMENT_B16>((__local_mem__ bfloat16_t*)(output) + offset, oBf16,
                                                                preg);
    } else {
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)(output) + offset, src, preg);
    }
}
}  // namespace ReduceOpTmpl
#endif  // _REDUCE_VAR_VF_COMMON_H_