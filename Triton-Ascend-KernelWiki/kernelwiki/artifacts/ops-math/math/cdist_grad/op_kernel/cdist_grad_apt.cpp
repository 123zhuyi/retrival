/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_grad_apt.cpp
 * \brief CdistGrad kernel entry (arch35)
 *
 * normMode is a compile-time template parameter — each binary contains only ONE DAG.
 * TilingKey encodes normMode so the framework loads the correct binary at runtime.
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/cdist_grad_dag.h"
#include "arch35/cdist_grad_tiling_key.h"
#include "cdist_grad_tiling_data.h"

using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

#define CDIST_GRAD_LAUNCH(DagT, ...)                                            \
    using _CGOp = ReduceSch<REDUCE_TPL_VALUE, DagT::OpDag>;                    \
    _CGOp _cgOp(&tilingData.reduceTiling);                                      \
    _cgOp.Init(&pipe, __VA_ARGS__);                                             \
    _cgOp.Process(static_cast<DTYPE_GRAD>(0))

template <REDUCE_TPL_PARAM, int32_t normMode>
__global__ __aicore__ void cdist_grad(
    GM_ADDR grad, GM_ADDR x1, GM_ADDR x2, GM_ADDR cdist,
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

    REGISTER_TILING_DEFAULT(CdistGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(CdistGradTilingData, tilingData, tiling);

    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_GRAD>::T;

    if constexpr (normMode == CdistGrad::NORM_MODE_INF) {
        using Dag = CdistGrad::CdistGradInfDag<DTYPE_GRAD, PromoteType>;
        CDIST_GRAD_LAUNCH(Dag, grad, x1, x2, cdist, y, userWS);
    } else if constexpr (normMode == CdistGrad::NORM_MODE_LARGE_P) {
        using OpLp = ReduceSch<REDUCE_TPL_VALUE,
            CdistGrad::CdistGradLargePDag<DTYPE_GRAD, PromoteType>::OpDag>;
        OpLp opLp(&tilingData.reduceTiling);
        opLp.template SetVar<PromoteType, 0>(static_cast<PromoteType>(tilingData.powCdist));
        opLp.Init(&pipe, grad, x1, x2, cdist, y, userWS);
        opLp.Process(static_cast<DTYPE_GRAD>(0));
    } else if constexpr (normMode == CdistGrad::NORM_MODE_P0) {
        using OpP0 = ReduceSch<REDUCE_TPL_VALUE,
            CdistGrad::CdistGradP0Dag<DTYPE_GRAD, PromoteType>::OpDag>;
        OpP0 opP0(&tilingData.reduceTiling);
        opP0.template SetVar<PromoteType, 0>(static_cast<PromoteType>(0.0f));
        // P0 DAG uses only In0, Out0 — no In1/In2/In3
        opP0.Init(&pipe, grad, y, userWS);
        opP0.Process(static_cast<DTYPE_GRAD>(0));
    } else if constexpr (normMode == CdistGrad::NORM_MODE_P1) {
        using Dag = CdistGrad::CdistGradP1Dag<DTYPE_GRAD, PromoteType>;
        CDIST_GRAD_LAUNCH(Dag, grad, x1, x2, y, userWS);
    } else if constexpr (normMode == CdistGrad::NORM_MODE_P2) {
        using Dag = CdistGrad::CdistGradP2Dag<DTYPE_GRAD, PromoteType>;
        CDIST_GRAD_LAUNCH(Dag, grad, x1, x2, cdist, y, userWS);
    } else {
        // NORM_MODE_GENERAL: 0 < p < 2, p != 1
        using Op = ReduceSch<REDUCE_TPL_VALUE,
            CdistGrad::CdistGradDag<DTYPE_GRAD, PromoteType>::OpDag>;
        Op op(&tilingData.reduceTiling);
        op.template SetVar<PromoteType, 0>(static_cast<PromoteType>(tilingData.powCdist));
        op.Init(&pipe, grad, x1, x2, cdist, y, userWS);
        op.Process(static_cast<DTYPE_GRAD>(0));
    }
}

#undef CDIST_GRAD_LAUNCH
