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
 * \file moe_re_routing_apt.cpp
 * \brief
 */

#include "arch35/moe_re_routing_re_regbase.h"
#include "arch35/moe_re_routing_r_regbase.h"
#include "kernel_operator.h"

using namespace MoeReRouting;

#define MOE_RE_ROUTING_RE_WITHOUT_SCALE 200000UL
#define MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT 200100UL
#define MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT8_E8M0 200200UL
#define MOE_RE_ROUTING_R_WITHOUT_SCALE 210000UL
#define MOE_RE_ROUTING_R_WITH_SCALE_FLOAT 210100UL
#define MOE_RE_ROUTING_R_WITH_SCALE_FLOAT8_E8M0 210200UL

extern "C" __global__ __aicore__ void moe_re_routing(GM_ADDR tokens, GM_ADDR expertTokenNumPerRank,
    GM_ADDR perTokenScales, GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales, GM_ADDR permuteTokenIdx,
    GM_ADDR expertTokenNum, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    if (TILING_KEY_IS(MOE_RE_ROUTING_RE_WITHOUT_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingReTilingData, tiling_data_in, tiling);
        const MoeReRoutingReTilingData *__restrict tilingData = &tiling_data_in;
        if constexpr (!IsSameType<DTYPE_TOKENS, fp8_e5m2_t>::value && !IsSameType<DTYPE_TOKENS, fp8_e4m3fn_t>::value) {
            MoeReRoutingReRegbase<DTYPE_TOKENS, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, false> op(&pipe, tilingData);
            op.Init(tokens,
                expertTokenNumPerRank,
                perTokenScales,
                permuteTokens,
                permutePerTokenScales,
                permuteTokenIdx,
                expertTokenNum);
            op.Process();
        }
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingReTilingData, tiling_data_in, tiling);
        const MoeReRoutingReTilingData *__restrict tilingData = &tiling_data_in;
        if constexpr (!IsSameType<DTYPE_TOKENS, fp8_e5m2_t>::value && !IsSameType<DTYPE_TOKENS, fp8_e4m3fn_t>::value) {
            MoeReRoutingReRegbase<DTYPE_TOKENS, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, true> op(&pipe, tilingData);
            op.Init(tokens,
                expertTokenNumPerRank,
                perTokenScales,
                permuteTokens,
                permutePerTokenScales,
                permuteTokenIdx,
                expertTokenNum);
            op.Process();
        }
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT8_E8M0)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingReTilingData, tiling_data_in, tiling);
        const MoeReRoutingReTilingData *__restrict tilingData = &tiling_data_in;
        MoeReRoutingReRegbase<int8_t, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, int8_t, true> op(&pipe, tilingData);
        op.Init(tokens,
            expertTokenNumPerRank,
            perTokenScales,
            permuteTokens,
            permutePerTokenScales,
            permuteTokenIdx,
            expertTokenNum);
        op.Process();
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_R_WITHOUT_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingRTilingData, tiling_data_in, tiling);
        const MoeReRoutingRTilingData *__restrict tilingData = &tiling_data_in;
        if constexpr (!IsSameType<DTYPE_TOKENS, fp8_e5m2_t>::value && !IsSameType<DTYPE_TOKENS, fp8_e4m3fn_t>::value) {
            MoeReRoutingRRegbase<DTYPE_TOKENS, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, false> op(&pipe, tilingData);
            op.Init(tokens,
                expertTokenNumPerRank,
                perTokenScales,
                permuteTokens,
                permutePerTokenScales,
                permuteTokenIdx,
                expertTokenNum);
            op.Process();
        }
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_R_WITH_SCALE_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingRTilingData, tiling_data_in, tiling);
        const MoeReRoutingRTilingData *__restrict tilingData = &tiling_data_in;
        if constexpr (!IsSameType<DTYPE_TOKENS, fp8_e5m2_t>::value && !IsSameType<DTYPE_TOKENS, fp8_e4m3fn_t>::value) {
            MoeReRoutingRRegbase<DTYPE_TOKENS, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, true> op(&pipe, tilingData);
            op.Init(tokens,
                expertTokenNumPerRank,
                perTokenScales,
                permuteTokens,
                permutePerTokenScales,
                permuteTokenIdx,
                expertTokenNum);
            op.Process();
        }
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_R_WITH_SCALE_FLOAT8_E8M0)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingRTilingData, tiling_data_in, tiling);
        const MoeReRoutingRTilingData *__restrict tilingData = &tiling_data_in;
        MoeReRoutingRRegbase<int8_t, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, int8_t, true> op(&pipe, tilingData);
        op.Init(tokens,
            expertTokenNumPerRank,
            perTokenScales,
            permuteTokens,
            permutePerTokenScales,
            permuteTokenIdx,
            expertTokenNum);
        op.Process();
    }
}