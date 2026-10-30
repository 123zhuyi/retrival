/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_min.cpp
 * \brief reduce_min
 */
#include "atvoss/reduce/reduce_sch.h"
#include "arch35/reduce_min_dag.h"
#include "arch35/reduce_min_tiling_key.h"

using namespace Ops::Base::ReduceOpTmpl;;
using namespace AscendC;

template <typename T>
__aicore__ inline constexpr T GetDumpValue()
{
    if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value) {
        return BFLOAT16_MAX_VALUE;
    } else if constexpr (IsSameType<DTYPE_X, half>::value) {
        return HALF_MAX_VALUE;
    } else if constexpr (IsSameType<DTYPE_X, float>::value) {
        return FLOAT_MAX_VALUE;
    } else if constexpr (IsSameType<DTYPE_X, uint8_t>::value) {
        return UINT8_MAX_VALUE;
    } else if constexpr (IsSameType<DTYPE_X, int8_t>::value) {
        return INT8_MAX_VALUE;
    } else if constexpr (IsSameType<DTYPE_X, int32_t>::value) {
        return INT32_MAX;
    } else if constexpr (IsSameType<DTYPE_X, int64_t>::value) {
        return INT64_MAX;
    }
}

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void reduce_min(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(ReduceOpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceOpTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value) {
        using Op = ReduceSch<REDUCE_TPL_VALUE, ReduceMin::ReduceMinBf16Dag<DTYPE_X, float>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, x, y, userWS);
        op.Process(GetDumpValue<DTYPE_X>());
    } else {
        using Op = ReduceSch<REDUCE_TPL_VALUE, ReduceMin::ReduceMinDag<DTYPE_X>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, x, y, userWS);
        op.Process(GetDumpValue<DTYPE_X>());
    }
}