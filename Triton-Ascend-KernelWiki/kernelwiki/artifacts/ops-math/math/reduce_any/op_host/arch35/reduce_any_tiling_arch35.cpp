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
 * \file reduce_any_tiling.cc
 * \brief tiling for reduce any
 */

#include <vector>
#include "math/reduce_any/op_kernel/arch35/reduce_any_dag.h"
#include "math/reduce_any/op_kernel/arch35/reduce_any_tiling_key.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {
using namespace Ops::Base;
static ge::graphStatus DoTiling(gert::TilingContext* context, ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto dType = xDesc->GetDataType();
    if (dType == ge::DT_FLOAT) {
        status = Tiling4ReduceOp<ReduceAny::ReduceAnyDagB32<float, int32_t>::OpDag>(context, opInput, key);
    } else {
        status = Tiling4ReduceOp<ReduceAny::ReduceAnyDag<int8_t, half>::OpDag>(context, opInput, key);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(dType).c_str(), "bool"),
        return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus Tiling4ReduceAny(gert::TilingContext* context)
{
    auto compileInfo = static_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_IF(
        compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "CompileInfo is nullptr"), return ge::GRAPH_FAILED);

    ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (ReduceOpTmpl::GetInputParam(context, opInput, 0, 1, 0) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "ReduceOp get x input param failed"), return ge::GRAPH_FAILED);

    ReduceTilingKey key;
    OP_CHECK_IF(
        (DoTiling(context, opInput, key) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "Tiling For ReduceAny Failed"), return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(
        context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, isContiguous:%d, Tiling Key is:%lu",
        key.patternID, key.loopARCount, key.loopInnerARCount, key.isContiguous ? 1 : 0, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReduceAny(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReduceAny).Tiling(Tiling4ReduceAny).TilingParse<ReduceOpCompileInfo>(TilingPrepare4ReduceAny);
} // namespace optiling