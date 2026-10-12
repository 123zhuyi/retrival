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
 * \file group_norm_silu_quant.cpp
 * \brief GroupNormSiluQuant arch35(Ascend950) regbase kernel entry.
 *        仅 same-type(x/gamma/beta 同 dtype); 无 mix-type。
 */

#include "group_norm_silu_quant_regbase_empty_tensor.h"
#include "group_norm_silu_quant_regbase_r_partial_load.h"
#include "group_norm_silu_quant_regbase_r_full_load.h"
#include "group_norm_silu_quant_regbase_r_partial_load_generalized.h"
#include "group_norm_silu_quant_regbase_r_full_load_generalized.h"
#include "group_norm_silu_quant_regbase_split_reduce.h"
#include "group_norm_silu_quant_regbase_many_tiny_groups.h"

using namespace GroupNormSiluQuant;

namespace {
#define TILINGKEY_EMPTY_TENSOR 1000
#define TILINGKEY_R_PARTIAL_LOAD 1100
#define TILINGKEY_R_FULL_LOAD 1110
#define TILINGKEY_R_PARTIAL_LOAD_GENERALIZED 1120
#define TILINGKEY_R_FULL_LOAD_GENERALIZED 1130
#define TILINGKEY_R_SPLIT_REDUCE 1140
#define TILINGKEY_MANY_TINY_GROUPS 1150
} // namespace

extern "C" __global__ __aicore__ void group_norm_silu_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale,
                                                            GM_ADDR yOut, GM_ADDR meanOut, GM_ADDR rstdOut,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(GroupNormSiluQuantRegbaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupNormSiluQuantRegbaseTilingData, tilingDataIn, tiling);
    const GroupNormSiluQuantRegbaseTilingData* __restrict tilingData = &tilingDataIn;
    if (TILING_KEY_IS(TILINGKEY_R_PARTIAL_LOAD)) {
        GroupNormSiluQuant::GroupNormSiluQuantRPartialLoad<DTYPE_X, DTYPE_X, false> op;
        op.Init(x, gamma, beta, quantScale, yOut, meanOut, rstdOut, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_R_FULL_LOAD)) {
        GroupNormSiluQuant::GroupNormSiluQuantRFullLoad<DTYPE_X, DTYPE_X, false> op;
        op.Init(x, gamma, beta, quantScale, yOut, meanOut, rstdOut, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_R_PARTIAL_LOAD_GENERALIZED)) {
        GroupNormSiluQuant::GroupNormSiluQuantRPartialLoad<DTYPE_X, DTYPE_X, true> op;
        op.Init(x, gamma, beta, quantScale, yOut, meanOut, rstdOut, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_R_FULL_LOAD_GENERALIZED)) {
        GroupNormSiluQuant::GroupNormSiluQuantRFullLoad<DTYPE_X, DTYPE_X, true> op;
        op.Init(x, gamma, beta, quantScale, yOut, meanOut, rstdOut, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_R_SPLIT_REDUCE)) {
        GroupNormSiluQuant::GroupNormSiluQuantSplitReduce<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, quantScale, yOut, meanOut, rstdOut, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_MANY_TINY_GROUPS)) {
        GroupNormSiluQuant::GroupNormSiluQuantManyTinyGroups<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, quantScale, yOut, meanOut, rstdOut, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_EMPTY_TENSOR)) {
        GroupNormSiluQuant::GroupNormSiluQuantEmpty<DTYPE_X> op;
        op.Init(x, gamma, beta, quantScale, yOut, meanOut, rstdOut, userWS, tilingData);
        op.Process();
    }
}
