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
 * \file reduce_mean_with_count_apt.cpp
 * \brief ReduceMeanWithCount kernel entry (arch35)
 *
 * Prototype: INPUT(x, count, count_sum) -> OUTPUT(y), ATTR(axes, keep_dims)
 * Computation (matches TBE): y = ReduceSum(x * count / count_sum, axes)
 * The DAG embeds Mul(x, count) and Div(result, count_sum) before ReduceSumOp.
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/reduce_mean_with_count_dag.h"
#include "arch35/reduce_mean_with_count_tiling_key.h"
#include "reduce_mean_with_count_tiling_data.h"

using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void reduce_mean_with_count(
    GM_ADDR x, GM_ADDR count, GM_ADDR count_sum,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
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

    REGISTER_TILING_DEFAULT(ReduceMeanWithCountTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceMeanWithCountTilingData, tilingData, tiling);

    // DAG: CopyIn(x,count,count_sum) -> Cast -> Mul -> Div -> ReduceSumOp -> Cast -> CopyOut(y)
    // Computation: y = ReduceSum(x * count / count_sum, axes)
    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_X>::T;
    using Op = ReduceSch<REDUCE_TPL_VALUE,
        ReduceMeanWithCount::ReduceMeanWithCountDag<DTYPE_X, PromoteType>::OpDag>;
    Op op(&tilingData.reduceTiling);
    op.Init(&pipe, x, count, count_sum, y, userWS);
    op.Process(static_cast<DTYPE_X>(0));
}
