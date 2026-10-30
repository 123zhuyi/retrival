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
 * \file reduce_prod_apt.cpp
 * \brief reduce_prod
 */
#include "atvoss/reduce/reduce_sch.h"
#include "arch35/reduce_prod_dag.h"
#include "arch35/reduce_prod_tiling_key.h"

using namespace Ops::Base;
using namespace AscendC;

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void reduce_prod(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
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

    if constexpr (std::is_same<DTYPE_X, int8_t>::value || std::is_same<DTYPE_X, uint8_t>::value ||
                  std::is_same<DTYPE_X, bool>::value) {
        using Op = ReduceOpTmpl::ReduceSch<REDUCE_TPL_VALUE, ReduceProd::ReduceProdI8Dag<int8_t, int16_t>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, x, y, userWS);
        op.Process(static_cast<int8_t>(1));
    } else {
        using PromoteType = ReduceOpTmpl::__reduceType::GetPromoteType<DTYPE_X>::T;
        using Op = ReduceOpTmpl::ReduceSch<REDUCE_TPL_VALUE, ReduceProd::ReduceProdDag<DTYPE_X, PromoteType>::OpDag>;
        Op op(&tilingData);
        op.Init(&pipe, x, y, userWS);
        op.Process(static_cast<DTYPE_X>(1));
    }
}