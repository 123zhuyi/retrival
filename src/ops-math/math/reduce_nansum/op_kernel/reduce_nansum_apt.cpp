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
 * \file reduce_nansum_apt.cpp
 * \brief ReduceNansum kernel entry for ascend950 (arch35).
 *
 * Prototype: INPUT(x, axes) -> OUTPUT(y), ATTR(keep_dims)
 * Computation: y = ReduceSum(NaN->0 preprocessing, axes)
 *
 * DAG pipeline:
 *   CopyIn(x) -> Cast(T->PromteT) -> ReduceNansumOp -> Cast(PromteT->T) -> CopyOut(y)
 *
 * NaN detection and replacement is integrated into ReduceNansumOp's CopyIn,
 * which performs NaN->0 (Compare(x,x,EQ) + Select) on each data chunk in UB
 * before the reduction computation, eliminating separate DAG nodes.
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/reduce_nansum_dag.h"
#include "arch35/reduce_nansum_operator.h"
#include "arch35/reduce_nansum_tiling_key.h"

using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void reduce_nansum(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
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

    // DAG: CopyIn(x) -> Cast -> ReduceNansumOp (NaN->0 integrated) -> Cast -> CopyOut(y)
    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_X>::T;
    using Op = ReduceSch<REDUCE_TPL_VALUE,
        ReduceNansum::ReduceNansumDag<DTYPE_X, PromoteType>::OpDag>;
    Op op(&tilingData);
    op.Init(&pipe, x, y, userWS);
    op.Process();
}