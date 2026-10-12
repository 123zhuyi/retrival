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
 * \file pfa_matmul_policy.h
 * \brief
 */
#ifndef PFA_MATMUL_POLICY_H
#define PFA_MATMUL_POLICY_H
#include "copy_cube_in/pa_copy_cube_in/pa_copy_left_cube_in_norm.h"
#include "copy_cube_in/pa_copy_cube_in/pa_copy_right_cube_in_norm.h"
#include "copy_cube_in/pa_copy_cube_in/pa_copy_left_cube_in_norm_split_k.h"
#include "copy_cube_in/pa_copy_cube_in/mla_pa_copy_right_cube_in_split_k.h"
#include "copy_cube_in/pfa_copy_cube_in_norm/pfa_copy_left_cube_in_norm.h"
#include "copy_cube_in/pfa_copy_cube_in_norm/pfa_copy_right_cube_in_norm.h"
#include "copy_cube_in/pfa_copy_cube_in_norm/mla_copy_left_cube_in_split_k.h"
#include "copy_cube_in/pfa_copy_cube_in_norm/mla_copy_right_cube_in_split_k.h"
#include "copy_cube_in/pfa_copy_cube_in_norm/mla_copy_right_cube_in_split_n.h"
#include "copy_cube_in/pfa_copy_cube_in_dn/pfa_copy_left_cube_in_dn.h"
#include "copy_cube_in/pfa_copy_cube_in_dn/pfa_copy_right_cube1_in_dn.h"
#include "copy_cube_in/pfa_copy_cube_in_dn/pfa_copy_right_cube2_in_dn.h"
#include "copy_cube_in/fa_pa_copy_cube_in/fa_pa_mm1_copy_left.h"
#include "copy_cube_in/fa_pa_copy_cube_in/fa_pa_mm1_copy_right.h"
#include "copy_cube_in/fa_pa_copy_cube_in/fa_pa_mm2_copy_right.h"
#include "cube_in_buffer/pfa_cube_in_buffer.h"
#include "pfa_policy_data.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class MatmulPolicyPa : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = PAFlagData;
    using CopyCubeInA = AscendC::Impl::Detail::PACopyLeftCubeInNorm<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = AscendC::Impl::Detail::PACopyRightCubeInNorm<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class MatmulPolicyPaD512 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = PAFlagData;
    using CopyCubeInA = AscendC::Impl::Detail::PACopyLeftCubeInSplitK<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = AscendC::Impl::Detail::PACopyRightCubeInNorm<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class PFANormBmm1Policy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = PFAMatmulPolicyData;
    using CopyCubeInA = AscendC::Impl::Detail::PFACopyLeftCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = AscendC::Impl::Detail::PFACopyRightCubeIn<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CubeInBufferB = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class PFANormBmm2Policy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = PFAMatmulPolicyData;
    using CopyCubeInB = AscendC::Impl::Detail::PFACopyRightCubeIn<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CubeInBufferB = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class IFAMLABmm1Policy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = IFAMLAMatmulPolicyData;
    using CopyCubeInA = AscendC::Impl::Detail::MLACopyLeftCubeInSplitK<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG, UserDefDataType>;
    using CubeInBufferA = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = AscendC::Impl::Detail::MLACopyRightCubeInSplitK<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CubeInBufferB = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class IFAMLABmm2Policy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = PFAMatmulPolicyData;
    using CopyCubeInB = AscendC::Impl::Detail::MLACopyRightCubeInSplitN<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CubeInBufferB = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class PFABmm1PolicyDN : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = PFAMatmulPolicyData;
    using CopyCubeInA = AscendC::Impl::Detail::PFACopyLeftCubeInDN<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = AscendC::Impl::Detail::PFACopyRightCubeInDN<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CubeInBufferB = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CopyCubeOut = AscendC::Impl::Detail::CopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, McgShfMode::DUAL_DST_SPLIT_N>; // 控制bmm1的计算结果竖切分给两个veccore
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class PFABmm2PolicyDN : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = PFAMatmulPolicyData;
    using CopyCubeInB = AscendC::Impl::Detail::PFACopyBmm2RightCubeInDN<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CubeInBufferB = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class IFAMLAPaBmm1Policy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = IFAMLAPaMatmulPolicyData;
    using CopyCubeInA = AscendC::Impl::Detail::MLACopyLeftCubeInSplitK<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG, UserDefDataType>;
    using CubeInBufferA = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = AscendC::Impl::Detail::MLAPaCopyRightCubeInSplitK<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class IFAMLAPaBmm2Policy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = PAFlagData;
    using CopyCubeInB = AscendC::Impl::Detail::PACopyRightCubeInNorm<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

// IFA(PFA) PA
template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FAPaBmm1Policy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FaPaPolicyData;
    using CopyCubeInA = AscendC::Impl::Detail::FaPaMm1CopyLeft<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = AscendC::Impl::Detail::FaPaMm1CopyRight<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CubeInBufferB = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FAPaBmm2Policy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FaPaPolicyData;
    using CopyCubeInB = AscendC::Impl::Detail::FaPaMm2CopyRight<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CubeInBufferB = AscendC::Impl::Detail::PFACubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC

#endif // PFA_MATMUL_POLICY_H