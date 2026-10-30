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
 * \file square_sum_v1_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../../op_kernel/square_sum_v1_dag.h"
#include "../../op_kernel/square_sum_v1_tiling_key.h"
#include "atvoss/reduce/reduce_tiling.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling {
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;

static ge::graphStatus DoTilingAscendC(gert::TilingContext* context, ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<SquareSumV1::SquareSumV1Dag<float, float>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<SquareSumV1::SquareSumV1Dag<half, float>::OpDag>(context, opInput, key);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "x", Ops::Base::ToString(opInput.inputDtype).c_str(), "bfloat16, float16 or float"),
        return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus Tiling4SquareSumV1AscendC(gert::TilingContext* context)
{
    ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (ReduceOpTmpl::GetInputParam(context, opInput, 0) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "ReduceOp get x input failed"), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto axis = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    auto axisData = static_cast<const int64_t*>(axis->GetData());
    if (axis->GetSize() == 0) {
        for (size_t i = 0; i < opInput.shape.size(); i++) {
            opInput.axes.push_back(i);
        }
    } else {
        size_t size = axis->GetSize();
        for (size_t i = 0; i < size; i++) {
            opInput.axes.push_back(axisData[i]);
        }
    }
    ReduceTilingKey key;
    OP_CHECK_IF(
        (DoTilingAscendC(context, opInput, key) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "DoTiling Failed for SquareSumV1"), return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(
        context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu", key.patternID,
        key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4SquareSumV1(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return Tiling4SquareSumV1AscendC(context);
}

static ge::graphStatus TilingPrepareForSquareSumV1([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the SquareSumV1 op.
IMPL_OP_OPTILING(SquareSumV1).Tiling(Tiling4SquareSumV1).TilingParse<ReduceOpCompileInfo>(TilingPrepareForSquareSumV1);
} // namespace optiling
