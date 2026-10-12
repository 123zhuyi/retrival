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
 * \file kl_div_v2.cpp
 * \brief kl_div_v2
 */

#include "atvoss/reduce/reduce_sch.h"
#include "kl_div_v2_dag.h"
#include "kl_div_v2_tiling_key.h"
#include "kl_div_v2_tiling_data.h"

using namespace Ops::Base;
using namespace ReduceOpTmpl;
using namespace AscendC;

enum Reduction
{
    None = 0,
    Mean = 1,
    Sum = 2,
    Batchmean = 3
};

enum LogTarget
{
    False = 0,
    True = 1
};

template <REDUCE_TPL_PARAM, int32_t reduction, int32_t logTarget>
__global__ __aicore__ void kl_div_v2(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KLDivV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(KLDivV2TilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (reduction == Reduction::Mean || reduction == Reduction::Batchmean) {
        if constexpr (logTarget == LogTarget::True) {
            using Op =
                ReduceSch<REDUCE_TPL_VALUE, KLDivV2::KLDivDagMeanLogTrue<DTYPE_X, float>::OpDag>;
            Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
            op.template SetVar<float, 0>(tilingData.reduceTiling.meanVar);
            op.Init(&pipe, x, target, y, workspace);
            op.Process(tilingData.emptyValue);
        } else {
            using Op =
                ReduceSch<REDUCE_TPL_VALUE, KLDivV2::KLDivDagMeanLogFalse<DTYPE_X, float>::OpDag>;
            Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
            op.template SetVar<float, 0>(tilingData.reduceTiling.meanVar);
            op.Init(&pipe, x, target, y, workspace);
            op.Process(tilingData.emptyValue);
        }
    } else if constexpr (reduction == Reduction::Sum || reduction == Reduction::None) {
        if constexpr (logTarget == LogTarget::True) {
            using Op =
                ReduceSch<REDUCE_TPL_VALUE, KLDivV2::KLDivDagSumLogTrue<DTYPE_X, float>::OpDag>;
            Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
            op.Init(&pipe, x, target, y, workspace);
            op.Process(tilingData.emptyValue);
        } else {
            using Op =
                ReduceSch<REDUCE_TPL_VALUE, KLDivV2::KLDivDagSumLogFalse<DTYPE_X, float>::OpDag>;
            Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
            op.Init(&pipe, x, target, y, workspace);
            op.Process(tilingData.emptyValue);
        }
    }
}