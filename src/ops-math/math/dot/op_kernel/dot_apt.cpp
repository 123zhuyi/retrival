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
 * \file dot.cpp
 * \brief kernel for dot
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/dot_dag.h"
#include "arch35/dot_tiling_key.h"

using namespace Ops::Base::ReduceOpTmpl;

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void dot(GM_ADDR input_x, GM_ADDR input_y, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
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

    if constexpr (IsSameType<DTYPE_INPUT_X, int8_t>::value || IsSameType<DTYPE_INPUT_X, uint8_t>::value) {
        using Op = ReduceSch<REDUCE_TPL_VALUE, Dot::DotDagI8<int8_t, int32_t>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, input_x, input_y, output, userWS);
        op.Process();
    } else {
        using PromoteType = __reduceType::GetPromoteType<DTYPE_INPUT_X>::T;
        using Op = ReduceSch<REDUCE_TPL_VALUE, Dot::DotDag<DTYPE_INPUT_X, PromoteType>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, input_x, input_y, output, userWS);
        op.Process();
    }
}