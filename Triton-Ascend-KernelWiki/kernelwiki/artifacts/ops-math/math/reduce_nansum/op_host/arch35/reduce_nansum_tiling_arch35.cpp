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
 * \file reduce_nansum_tiling_arch35.cpp
 * \brief Tiling for ReduceNansum on ascend950 (arch35).
 *
 * Prototype: INPUT(x, axes) -> OUTPUT(y), ATTR(keep_dims)
 * Computation: y = ReduceSum(Select(Compare(x, x, EQ), x, 0), axes)
 * The DAG embeds NaN->0 preprocessing before ReduceSumOp.
 */

#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "op_host/tiling_base_util.h"
#include "math/reduce_nansum/op_kernel/arch35/reduce_nansum_dag.h"
#include "math/reduce_nansum/op_kernel/arch35/reduce_nansum_tiling_key.h"

using namespace Ops::Base;

namespace optiling {
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
static ge::graphStatus DoTiling(gert::TilingContext* context, ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    // Debug: print input parameters before tiling
    OP_LOGI(
        context->GetNodeName(),
        "[DEBUG] Input: dtype:%d, shape size:%zu, axes size:%zu, shape:[%s], strides:[%s], axes:[%s]",
        static_cast<int>(opInput.inputDtype), opInput.shape.size(), opInput.axes.size(),
        ReduceOpTmpl::VectorToString(opInput.shape).c_str(),
        ReduceOpTmpl::VectorToString(opInput.dimStrides).c_str(),
        ReduceOpTmpl::VectorToString(opInput.axes).c_str());

    ge::graphStatus status = ge::GRAPH_FAILED;

    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<ReduceNansum::ReduceNansumDag<float, float>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<ReduceNansum::ReduceNansumDag<half, float>::OpDag>(context, opInput, key);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE(
            context->GetNodeName(), "ReduceOp Tiling failed, dtype shoude be in (bfloat16/float16/float)"),
        return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus Tiling4ReduceNansum(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (ReduceOpTmpl::GetInputParam(context, opInput, 0, 1, 0) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "ReduceOp get x input param failed"), return ge::GRAPH_FAILED);

    if (opInput.axes.empty()) {
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const bool isNoopWithEmpty = *(attrs->GetAttrPointer<bool>(1));
        if (!isNoopWithEmpty) {
            opInput.axes.resize(opInput.shape.size());
            for (size_t i = 0; i < opInput.shape.size(); i++) {
                opInput.axes[i] = i;
            }
        }
    }

    ReduceTilingKey key;
    OP_CHECK_IF(
        (DoTiling(context, opInput, key) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "DoTiling Failed for ReduceNansum"), return ge::GRAPH_FAILED);

    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(
        context->GetNodeName(),
        "patternID:%u, loopARCount:%u, loopInnerARCount:%u, isContiguous:%d, Tiling Key is:%lu",
        key.patternID, key.loopARCount, key.loopInnerARCount, key.isContiguous ? 1 : 0, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReduceNansum(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReduceNansum).Tiling(Tiling4ReduceNansum).TilingParse<ReduceOpCompileInfo>(TilingPrepare4ReduceNansum);
}  // namespace optiling