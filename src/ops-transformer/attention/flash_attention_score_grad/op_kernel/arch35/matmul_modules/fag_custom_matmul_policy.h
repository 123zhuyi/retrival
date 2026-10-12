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
 * \file fag_custom_matmul_policy.h
 * \brief
 */
#ifndef FAG_CUSTOM_MATMUL_POLICY_H
#define FAG_CUSTOM_MATMUL_POLICY_H

#include "copy_cube_in/copy_cube_in_s1s2_align/fag_copy_cube_in_post.h"
#include "copy_cube_in/copy_cube_in_s1s2_align/fag_copy_cube_in_pre.h"
#include "copy_cube_in/copy_cube_in_s1s2_preload/fag_copy_cube_in_post_preload.h"
#include "copy_cube_in/copy_cube_in_s1s2_preload/fag_copy_cube_in_pre_preload.h"
#include "copy_cube_in/copy_cube_in_s1s2_rope/fag_copy_cube_in_pre_rope.h"
#include "cube_in_buffer/fag_cube_in_buffer_post.h"
#include "cube_in_buffer/fag_cube_in_buffer_pre.h"
#include "cube_out_buffer/fag_cube_out_buffer_mm1.h"
#include "cube_out_buffer/fag_cube_out_buffer_mm3.h"
#include "lib/../../impl/adv_api/detail/matmul/policy/matmul_policy.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG> class FagContextS1S2Align
{
    using SrcT = typename INPUT_TYPE::T;

public:
    __aicore__ inline FagContextS1S2Align() {}
    __aicore__ inline void Init() {}
    uint8_t leftMatrixEncodingTableIdx;
    uint8_t rightMatrixEncodingTableIdx;
    uint64_t loopIndex = 0;
};

// no preload
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2MM1 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2Pre<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2Pre<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Pre<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Pre<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2MM2 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2MM3 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

// no preload + L0c
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2L0cMM1 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2Pre<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2Pre<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Pre<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Pre<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeOutBuffer = FagCubeOutBufferMM1<IMPL, MatmulInputCType<C_TYPE, typename C_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2L0cMM2 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeOutBuffer = FagCubeOutBufferMM1<IMPL, MatmulInputCType<C_TYPE, typename C_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2L0cMM3 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeOutBuffer = FagCubeOutBufferMM3<IMPL, MatmulInputCType<C_TYPE, typename C_TYPE::T>, MM_CFG>;
};

// with preload
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG> class FagContextS1S2Preload
{
    using SrcT = typename INPUT_TYPE::T;

public:
    __aicore__ inline void Init() {}
    uint8_t leftMatrixEncodingTableIdx;
    uint8_t rightMatrixEncodingTableIdx;
    event_t eventIDMte2ToMte1MM1;
    event_t eventIDMte2ToMte1MM2;
    event_t eventIDTable[TSCM_BUF_NUM];
    uint64_t loopIndex = 0;
    __aicore__ inline FagContextS1S2Preload()
    {
        eventIDMte2ToMte1MM1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        eventIDMte2ToMte1MM2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        eventIDTable[0] = eventIDMte2ToMte1MM1; // dx ping
        eventIDTable[2] = eventIDMte2ToMte1MM1; // dx pong
        eventIDTable[7] = eventIDMte2ToMte1MM1; // dx preload
        eventIDTable[3] = eventIDMte2ToMte1MM2; // q ping
        eventIDTable[5] = eventIDMte2ToMte1MM2; // q pong
        eventIDTable[8] = eventIDMte2ToMte1MM2; // q preload
    }
    __aicore__ inline ~FagContextS1S2Preload()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1MM1);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIDMte2ToMte1MM2);
    }
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2PreloadMM1 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2PrePreload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2PrePreload<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Pre<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Pre<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Preload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2RopePreloadMM1 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmRopeFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2PreRopePreload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2PreRopePreload<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Pre<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Pre<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Preload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2PreloadMM2 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA =
        FagCopyCubeInS1S2PostPreload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB =
        FagCopyCubeInS1S2PostPreload<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2PreloadMM3 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA =
        FagCopyCubeInS1S2PostPreload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB =
        FagCopyCubeInS1S2PostPreload<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

// with preload + l0c
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2PreloadL0cMM1 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA = FagCopyCubeInS1S2PrePreload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInS1S2PrePreload<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Pre<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Pre<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Preload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeOutBuffer = FagCubeOutBufferMM1<IMPL, MatmulInputCType<C_TYPE, typename C_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2PreloadL0cMM2 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA =
        FagCopyCubeInS1S2PostPreload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB =
        FagCopyCubeInS1S2PostPreload<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeOutBuffer = FagCubeOutBufferMM1<IMPL, MatmulInputCType<C_TYPE, typename C_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyS1S2PreloadL0cMM3 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using UserDefDataType = FagTscmFlagData;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG, UserDefDataType>;
    using CopyCubeInA =
        FagCopyCubeInS1S2PostPreload<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB =
        FagCopyCubeInS1S2PostPreload<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferS1S2Post<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferS1S2Post<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContextS1S2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeOutBuffer = FagCubeOutBufferMM3<IMPL, MatmulInputCType<C_TYPE, typename C_TYPE::T>, MM_CFG>;
};

}
}
}

#endif