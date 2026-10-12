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
 * \file reduce_all.cpp
 * \brief reduce_all
 */
#include "atvoss/reduce/reduce_sch.h"
#include "./arch35/reduce_all_dag.h"
#include "./arch35/reduce_all_tiling_key.h"

using namespace Ops::Base;
using namespace ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void reduce_all(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
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
    if constexpr (IsSameType<DTYPE_X, float>::value) {
        using Op = ReduceSch<REDUCE_TPL_VALUE, ReduceAll::ReduceAllDagFloat<DTYPE_X, int32_t>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, x, y, userWS);
        op.Process(static_cast<uint8_t>(1));
    } else if constexpr(IsSameType<DTYPE_X, half>::value||IsSameType<DTYPE_X, bfloat16_t>::value) {
        using Op = ReduceSch<REDUCE_TPL_VALUE, ReduceAll::ReduceAllDagFloat<DTYPE_X, int16_t>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, x, y, userWS);
        op.Process(static_cast<uint8_t>(1));
    } else {
        using PromoteType = __reduceType::GetPromoteType<uint8_t>::T;
        using Op = ReduceSch<REDUCE_TPL_VALUE, ReduceAll::ReduceAllDag<uint8_t, PromoteType>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, x, y, userWS);
        op.Process(static_cast<uint8_t>(1));
    }
}