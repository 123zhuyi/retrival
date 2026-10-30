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
 * \file reduce_log_sum_exp_tiling.cc
 * \brief tiling for reduce_log_sum_exp
 */

#include "reduce_log_sum_exp_tiling_arch35.h"
#include <vector>
#include "math/reduce_log_sum_exp/op_kernel/arch35/reduce_log_sum_exp_dag.h"
#include "math/reduce_log_sum_exp/op_kernel/arch35/reduce_log_sum_exp_tiling_key.h"

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling
{
using namespace Ops::Base;
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
static ge::graphStatus DoTiling(gert::TilingContext* context, ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<ReduceLogSumExp::ReduceLogSumExpDag<float>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<ReduceLogSumExp::ReduceLogSumExpDag<half>::OpDag>(context, opInput, key);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "x", Ops::Base::ToString(opInput.inputDtype).c_str(), "bfloat16, float16 or float"),
        return ge::GRAPH_FAILED);
    return status;
}

ge::graphStatus Tiling4ReduceLogSumExp(gert::TilingContext* context)
{
    auto compileInfo = static_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "CompileInfo is nullptr"),
                    return ge::GRAPH_FAILED);

    ReduceOpInputParam opInput;
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(context, opInput, 0, 1, 0) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "ReduceOp get x input param failed"),
                    return ge::GRAPH_FAILED);
    ReduceTilingKey key;
    OP_CHECK_IF((DoTiling(context, opInput, key) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "DoTiling Failed for ReduceLogSumExp"),
                    return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu",
            key.patternID, key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ReduceLogSumExp(gert::TilingParseContext* context)
{
    (void) context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReduceLogSumExp)
    .Tiling(Tiling4ReduceLogSumExp)
    .TilingParse<ReduceOpCompileInfo>(TilingPrepare4ReduceLogSumExp)
    .TilingInputsDataDependency({1});
}  // namespace optiling