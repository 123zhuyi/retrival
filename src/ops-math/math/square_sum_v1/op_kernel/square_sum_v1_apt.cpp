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
 * \file square_sum_v1.cpp
 * \brief square_sum_v1
 */

#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/reduce/reduce_sch.h"
#include "square_sum_v1_dag.h"
#include "square_sum_v1_tiling_key.h"
#include "square_sum_v1_tiling_data.h"

using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM, uint32_t Noop>
__global__ __aicore__ void square_sum_v1(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
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
    REGISTER_TILING_DEFAULT(SquareSumV1TilingData);
    GET_TILING_DATA_WITH_STRUCT(SquareSumV1TilingData, tilingData, tiling);
    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_X>::T;
    if constexpr (Noop == 1) {
        ElementwiseSch<0UL, SquareSumV1::SquareSumV1NoopDag<DTYPE_X, PromoteType>::OpDag> sch(
            &(tilingData.elewiseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else {
        using Op = ReduceSch<REDUCE_TPL_VALUE, SquareSumV1::SquareSumV1Dag<DTYPE_X, PromoteType>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.Init(&pipe, x, y, userWS);
        op.Process();
    }
}
