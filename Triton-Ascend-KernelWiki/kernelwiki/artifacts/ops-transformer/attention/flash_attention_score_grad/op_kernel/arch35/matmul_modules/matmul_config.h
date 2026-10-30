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
 * \file matmul_config.h
 * \brief
 */
 
#ifndef MATMUL_CONFIG_H
#define MATMUL_CONFIG_H
#include "fag_custom_matmul_policy.h"

// BNS1S2模板
///////////////////////////////////////////////////////////////
// mm1 policy selector  
///////////////////////////////////////////////////////////////
using namespace AscendC::Impl::Detail;

template <bool policySwitch, bool tscmPreloadSwitch, bool l0cSwitch, bool ropeSwitch> struct Mm1ConstPolicySelector {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm1ConstPolicySelector<true, true, false, false> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2PreloadMM1<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm1ConstPolicySelector<true, false, false, false> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2MM1<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm1ConstPolicySelector<true, true, true, false> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2PreloadL0cMM1<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm1ConstPolicySelector<true, false, true, false> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2L0cMM1<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};

template <> struct Mm1ConstPolicySelector<true, true, false, true> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2RopePreloadMM1<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
///////////////////////////////////////////////////////////////
// mm2 policy selector
///////////////////////////////////////////////////////////////
template <bool policySwitch, bool tscmPreloadSwitch, bool l0cSwitch> struct Mm2ConstPolicySelector {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm2ConstPolicySelector<true, true, false> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2PreloadMM2<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm2ConstPolicySelector<true, false, false> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2MM2<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm2ConstPolicySelector<true, true, true> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2PreloadL0cMM2<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm2ConstPolicySelector<true, false, true> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2L0cMM2<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
 
///////////////////////////////////////////////////////////////
// mm3 policy selector
///////////////////////////////////////////////////////////////
template <bool policySwitch, bool tscmPreloadSwitch, bool l0cSwitch> struct Mm3ConstPolicySelector {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm3ConstPolicySelector<true, true, false> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2PreloadMM3<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm3ConstPolicySelector<true, false, false> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2MM3<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm3ConstPolicySelector<true, true, true> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2PreloadL0cMM3<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
template <> struct Mm3ConstPolicySelector<true, false, true> {
    template < const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2L0cMM3<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    };
};
 
///////////////////////////////////////////////////////////////
// mm1 static mm config
///////////////////////////////////////////////////////////////
template <typename T1>
__aicore__ inline constexpr MatmulConfig
GetMm1Cfg(const bool IS_ATTEN_MASK, const bool IS_PSE, const bool IS_DROP,
          const uint32_t CUBE_BASEM, const uint32_t CUBE_BASEN, const uint32_t HEAD_DIM_ALIGN, const bool IS_TSCM_REUSE,
          const bool IS_L0DB, const bool IS_L0C_REUSE, const uint32_t sharedCO1BufferSize, const uint32_t maxBaseRatio)
{
    MatmulShapeParams shapeParams = {CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN,
                                     CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN / maxBaseRatio};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    // 各个enale参数 根据算子根据选择，确认是否需要使能，不需要使能可置为false, MMAPI
    // 根据此配置进行对应分支的常量折叠，以达到极致性能
    mmCfg.intrinsicsCheck = false;
    mmCfg.enUnitFlag = false;
    mmCfg.enableInit = false;
    mmCfg.enableSetBias = false;
    mmCfg.enableQuantVector = false;
    mmCfg.enableSetTail = true;
    mmCfg.isBiasBatch = false;
    mmCfg.enableSetDefineData = IS_TSCM_REUSE || IS_L0C_REUSE;
    mmCfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
    mmCfg.isA2B2Shared = IS_L0DB;
    mmCfg.isCO1Shared = !IS_L0C_REUSE;
    mmCfg.sharedCO1BufferSize = sharedCO1BufferSize;
    return mmCfg;
}
 
///////////////////////////////////////////////////////////////
// mm2 static mm config
///////////////////////////////////////////////////////////////
template <typename T1>
__aicore__ inline constexpr MatmulConfig
GetMm2Cfg(const bool IS_ATTEN_MASK, const bool IS_PSE, const bool IS_DROP,
          const uint32_t CUBE_BASEM, const uint32_t CUBE_BASEN, const uint32_t HEAD_DIM_ALIGN, const bool IS_TSCM_REUSE,
          const bool IS_L0DB, const bool IS_L0C_REUSE, const uint32_t sharedCO1BufferSize, const uint32_t maxBaseRatio)
{
    MatmulShapeParams shapeParams = {CUBE_BASEM, HEAD_DIM_ALIGN, CUBE_BASEN,
                                     CUBE_BASEM, HEAD_DIM_ALIGN, CUBE_BASEN / maxBaseRatio};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    // 各个enale参数 根据算子根据选择，确认是否需要使能，不需要使能可置为false, MMAPI
    // 根据此配置进行对应分支的常量折叠，以达到极致性能
    mmCfg.intrinsicsCheck = false;
    mmCfg.enUnitFlag = false;
    mmCfg.enableInit = false;
    mmCfg.enableSetBias = false;
    mmCfg.enableQuantVector = IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value;
    mmCfg.enableSetTail = true;
    mmCfg.isBiasBatch = false;
    mmCfg.enableSetDefineData = IS_TSCM_REUSE || IS_L0C_REUSE;
    mmCfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
    mmCfg.enableStaticPadZeros = false;
    mmCfg.isA2B2Shared = IS_L0DB;
    mmCfg.isCO1Shared = !IS_L0C_REUSE;
    mmCfg.sharedCO1BufferSize = sharedCO1BufferSize;
    return mmCfg;
}
 
///////////////////////////////////////////////////////////////
// mm3 static mm config
///////////////////////////////////////////////////////////////
template <typename T1>
__aicore__ inline constexpr MatmulConfig
GetMm3Cfg(const bool IS_ATTEN_MASK, const bool IS_PSE, const bool IS_DROP,
          const uint32_t CUBE_BASEM, const uint32_t CUBE_BASEN, const uint32_t HEAD_DIM_ALIGN, const bool IS_TSCM_REUSE,
          const bool IS_L0DB, const bool IS_L0C_REUSE, const uint32_t sharedCO1BufferSize, const uint32_t maxBaseRatio)
{
    MatmulShapeParams shapeParams = {CUBE_BASEN, HEAD_DIM_ALIGN, CUBE_BASEM,
                                     CUBE_BASEN, HEAD_DIM_ALIGN, CUBE_BASEM / maxBaseRatio};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    // 各个enale参数 根据算子根据选择，确认是否需要使能，不需要使能可置为false, MMAPI
    // 根据此配置进行对应分支的常量折叠，以达到极致性能
    mmCfg.intrinsicsCheck = false;
    mmCfg.enUnitFlag = false;
    mmCfg.enableInit = IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value;
    mmCfg.enableSetBias = false;
    mmCfg.enableQuantVector = IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value;
    mmCfg.enableSetTail = true;
    mmCfg.isBiasBatch = false;
    mmCfg.enableSetDefineData = IS_TSCM_REUSE || IS_L0C_REUSE;
    mmCfg.iterateMode = (IS_L0C_REUSE) ? IterateMode::ITERATE_MODE_NORMAL : IterateMode::ITERATE_MODE_ALL;
    mmCfg.enableStaticPadZeros = false;
    mmCfg.isA2B2Shared = IS_L0DB;
    mmCfg.isCO1Shared = !IS_L0C_REUSE;
    mmCfg.sharedCO1BufferSize = sharedCO1BufferSize;
    return mmCfg;
}

// BN2模板
///////////////////////////////////////////////////////////////
// mm1 L1 policy selector
///////////////////////////////////////////////////////////////
template <bool policySwitch, bool tscmPreloadSwitch> struct Mm1ConstPolicyBN2Selector {
    template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct Mm1ConstPolicyBN2Selector<true, true> {
    template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2PreloadMM1<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct Mm1ConstPolicyBN2Selector<true, false> {
    template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2MM1<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

///////////////////////////////////////////////////////////////
// mm3 L1 policy selector
///////////////////////////////////////////////////////////////
template <bool policySwitch, bool tscmPreloadSwitch> struct Mm3ConstPolicyBN2Selector {
    template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct Mm3ConstPolicyBN2Selector<true, true> {
    template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2PreloadMM3<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct Mm3ConstPolicyBN2Selector<true, false> {
    template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyS1S2MM3<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

__aicore__ constexpr uint32_t BN2CeilConst(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ constexpr TPosition GetC2Position(DTemplateType dTemplateType)
{
    if ((uint16_t)dTemplateType <= (uint16_t)DTemplateType::Aligned128) {
        return TPosition::VECCALC;
    } else {
        return TPosition::GM;
    }
}

///////////////////////////////////////////////////////////////
// mm1 static mm config
///////////////////////////////////////////////////////////////
template <typename T1>
__aicore__ inline constexpr MatmulConfig GetBN2Mm1Cfg(const bool IS_ATTEN_MASK, const bool IS_PSE, const bool IS_DROP,
                                                      const bool HAS_TAIL, const uint32_t CUBE_BASEM,
                                                      const uint32_t CUBE_BASEN, const uint32_t HEAD_DIM_ALIGN,
                                                      const bool IS_TSCM_REUSE, const bool IS_L0DB,
                                                      const uint32_t sharedCO1BufferSize, const uint32_t maxBaseRatio)
{
    MatmulShapeParams shapeParams = {CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN,
                                     CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN / maxBaseRatio};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    // 各个enale参数 根据算子根据选择，确认是否需要使能，不需要使能可置为false, MMAPI
    // 根据此配置进行对应分支的常量折叠，以达到极致性能
    mmCfg.intrinsicsCheck = false;
    mmCfg.enUnitFlag = false;
    mmCfg.enableInit = false;
    mmCfg.enableSetBias = false;
    mmCfg.enableQuantVector = false;
    mmCfg.enableSetTail = HAS_TAIL;
    mmCfg.isBiasBatch = false;
    mmCfg.enableSetDefineData = IS_TSCM_REUSE;
    mmCfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
    mmCfg.isA2B2Shared = IS_L0DB;
    mmCfg.isCO1Shared = true;
    mmCfg.sharedCO1BufferSize = sharedCO1BufferSize;
    return mmCfg;
}

///////////////////////////////////////////////////////////////
// mm2 static mm config
///////////////////////////////////////////////////////////////
template <typename T1>
__aicore__ inline constexpr MatmulConfig GetBN2Mm2Cfg(const bool IS_ATTEN_MASK, const bool IS_PSE, const bool IS_DROP,
                                                      const bool HAS_TAIL, const uint32_t CUBE_BASEM,
                                                      const uint32_t CUBE_BASEN, const uint32_t HEAD_DIM_ALIGN,
                                                      const bool IS_TSCM_REUSE, const bool IS_L0DB,
                                                      const uint32_t sharedCO1BufferSize, const uint32_t maxBaseRatio)
{
    MatmulShapeParams shapeParams = {CUBE_BASEM, HEAD_DIM_ALIGN, CUBE_BASEN,
                                     CUBE_BASEM, HEAD_DIM_ALIGN, CUBE_BASEN / maxBaseRatio};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    // 各个enale参数 根据算子根据选择，确认是否需要使能，不需要使能可置为false, MMAPI
    // 根据此配置进行对应分支的常量折叠，以达到极致性能
    mmCfg.intrinsicsCheck = false;
    mmCfg.enUnitFlag = false;
    mmCfg.enableInit = false;
    mmCfg.enableSetBias = false;
    mmCfg.enableQuantVector = false;
    mmCfg.enableSetTail = HAS_TAIL;
    mmCfg.isBiasBatch = false;
    mmCfg.enableSetDefineData = IS_TSCM_REUSE;
    mmCfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
    mmCfg.enableStaticPadZeros = false;
    mmCfg.isA2B2Shared = IS_L0DB;
    mmCfg.isCO1Shared = true;
    mmCfg.sharedCO1BufferSize = sharedCO1BufferSize;
    return mmCfg;
}

///////////////////////////////////////////////////////////////
// mm3 static mm config
///////////////////////////////////////////////////////////////
template <typename T1>
__aicore__ inline constexpr MatmulConfig GetBN2Mm3Cfg(const bool IS_ATTEN_MASK, const bool IS_PSE, const bool IS_DROP,
                                                      const bool HAS_TAIL, const uint32_t CUBE_BASEM,
                                                      const uint32_t CUBE_BASEN, const uint32_t HEAD_DIM_ALIGN,
                                                      const bool IS_TSCM_REUSE, const bool IS_L0DB,
                                                      const uint32_t sharedCO1BufferSize, const uint32_t maxBaseRatio)
{
    MatmulShapeParams shapeParams = {CUBE_BASEN, HEAD_DIM_ALIGN, CUBE_BASEM,
                                     CUBE_BASEN, HEAD_DIM_ALIGN, CUBE_BASEM / maxBaseRatio};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    // 各个enale参数 根据算子根据选择，确认是否需要使能，不需要使能可置为false, MMAPI
    // 根据此配置进行对应分支的常量折叠，以达到极致性能
    mmCfg.intrinsicsCheck = false;
    mmCfg.enUnitFlag = false;
    mmCfg.enableInit = false;
    mmCfg.enableSetBias = false;
    mmCfg.enableQuantVector = false;
    mmCfg.enableSetTail = HAS_TAIL;
    mmCfg.isBiasBatch = false;
    mmCfg.enableSetDefineData = IS_TSCM_REUSE;
    mmCfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
    mmCfg.enableStaticPadZeros = false;
    mmCfg.isA2B2Shared = IS_L0DB;
    mmCfg.isCO1Shared = true;
    mmCfg.sharedCO1BufferSize = sharedCO1BufferSize;
    return mmCfg;
}
 
#endif