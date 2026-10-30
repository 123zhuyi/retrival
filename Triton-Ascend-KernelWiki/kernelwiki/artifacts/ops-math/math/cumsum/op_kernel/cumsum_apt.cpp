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
 * \file cumsum_apt.cpp
 * \brief calculate the prefix accumulation sum of tensor specified dimension
 */
#include "arch35/cumsum_4_int.h"
#include "arch35/cumsum_core_ss_oneway_ss.h"
#include "arch35/cumsum_core_ss_twoway_ss.h"
#include "arch35/cumsum_core_ss_ub_ss_oneway_ss.h"
#include "arch35/cumsum_core_ss_ub_ss_twoway_ss.h"
#include "arch35/cumsum_oneway_ss.h"
#include "arch35/cumsum_twoway_ss.h"
#include "arch35/cumsum_ub_ss_oneway_ss.h"
#include "arch35/cumsum_ub_ss_twoway_ss.h"

using namespace AscendC;
using namespace Cumsum;
using namespace Cum;

#define CUMSUM_ONEWAY_SS_TILING_KEY 1001
#define CUMSUM_TWOWAY_SS_TILING_KEY 1002
#define CUMSUM_UB_SS_ONEWAY_SS_TILING_KEY 1011
#define CUMSUM_UB_SS_TWOWAY_SS_TILING_KEY 1012
#define CUMSUM_CORE_SS_ONEWAY_SS_TILING_KEY 1101
#define CUMSUM_CORE_SS_TWOWAY_SS_TILING_KEY 1102
#define CUMSUM_CORE_SS_UB_SS_ONEWAY_SS_TILING_KEY 1111
#define CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_TILING_KEY 1112
#define CUM_NO_SPLIT 11000
#define CUM_AR_SPLIT 11001
#define CUM_WITH_GROUP 11002

extern "C" __aicore__ inline void cumsumSimd(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    using PromoteType = __cumsumType::GetPromoteType<DTYPE_X>::T;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(CUMSUM_ONEWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (
            std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
            std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumOnewaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_TWOWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (
            std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
            std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumTwowaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }

    } else if (TILING_KEY_IS(CUMSUM_UB_SS_ONEWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (
            std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
            std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumUbSsOnewaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_UB_SS_TWOWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (
            std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
            std::is_same<DTYPE_X, bfloat16_t>::value) {
            CumsumUbSsTwowaySs<DTYPE_X, PromoteType> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_CORE_SS_ONEWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (
            std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
            std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_CORE_SS_ONEWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsOnewaySs<DTYPE_X, PromoteType, CumsumOnewaySklansky<DTYPE_X, PromoteType>> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_CORE_SS_TWOWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (
            std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
            std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_CORE_SS_TWOWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsTwowaySs<DTYPE_X, PromoteType, CumsumTwowaySklansky<DTYPE_X, PromoteType>> op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_CORE_SS_UB_SS_ONEWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (
            std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
            std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_CORE_SS_UB_SS_ONEWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsUbSsOnewaySs<
                DTYPE_X, PromoteType,
                CumsumUbSklansky<DTYPE_X, PromoteType, CumsumOnewaySklansky<DTYPE_X, PromoteType>>>
                op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(CumsumSklanskyTilingData, tilingDataIn, tiling);
        const CumsumSklanskyTilingData* __restrict tilingData = &tilingDataIn;
        if constexpr (
            std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
            std::is_same<DTYPE_X, bfloat16_t>::value) {
            KERNEL_TASK_TYPE(CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
            CumsumCoreSsUbSsTwowaySs<
                DTYPE_X, PromoteType,
                CumsumUbSklansky<DTYPE_X, PromoteType, CumsumTwowaySklansky<DTYPE_X, PromoteType>>>
                op(pipe);
            op.Init(x, y, tilingData, workspace);
            op.Process();
        }
    }
}

extern "C" __aicore__ inline void cumsumSimdInt(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);

    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(CUM_NO_SPLIT)) {
        GET_TILING_DATA_WITH_STRUCT(Cum4IntTilingData, tilingDataInt, tiling);
        if constexpr (
            std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, int64_t>::value ||
            std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
            std::is_same<DTYPE_X, uint64_t>::value) {
            CumNoSplit<DTYPE_X> op;
            op.Init(x, y, &tilingDataInt, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUM_AR_SPLIT)) {
        GET_TILING_DATA_WITH_STRUCT(Cum4IntTilingData, tilingDataInt, tiling);
        if constexpr (
            std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, int64_t>::value ||
            std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
            std::is_same<DTYPE_X, uint64_t>::value) {
            CumSplitAR<DTYPE_X> op;
            op.Init(x, y, &tilingDataInt, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(CUM_WITH_GROUP)) {
        GET_TILING_DATA_WITH_STRUCT(Cum4IntTilingData, tilingDataInt, tiling);
        KERNEL_TASK_TYPE(CUM_WITH_GROUP, KERNEL_TYPE_MIX_AIV_1_0);
        if constexpr (
            std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, int64_t>::value ||
            std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
            std::is_same<DTYPE_X, uint64_t>::value) {
            CumWithGroup<DTYPE_X> op;
            op.Init(x, y, &tilingDataInt, &pipe);
            op.Process();
        }
    }
}

extern "C" __global__ __aicore__ void cumsum(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (
        std::is_same<DTYPE_X, half>::value || std::is_same<DTYPE_X, float>::value ||
        std::is_same<DTYPE_X, bfloat16_t>::value) {
        cumsumSimd(x, axis, y, workspace, tiling);
    } else if constexpr (
        std::is_same<DTYPE_X, int32_t>::value || std::is_same<DTYPE_X, int64_t>::value ||
        std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
        std::is_same<DTYPE_X, uint64_t>::value) {
        cumsumSimdInt(x, axis, y, workspace, tiling);
    }
}
