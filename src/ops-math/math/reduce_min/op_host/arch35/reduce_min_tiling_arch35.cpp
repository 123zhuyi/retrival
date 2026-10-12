/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_min_tiling_arch35.cpp
 * \brief tiling for reduce min
 */

#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "op_host/tiling_base_util.h"
#include "math/reduce_min/op_kernel/arch35/reduce_min_dag.h"
#include "math/reduce_min/op_kernel/arch35/reduce_min_tiling_key.h"

using namespace Ops::Base;

namespace optiling {
static constexpr int32_t SIZE8 = 8;
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
static ge::graphStatus DoTiling(gert::TilingContext* context, ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (opInput.inputDtype == ge::DT_BF16) {
        status = Tiling4ReduceOp<ReduceMin::ReduceMinBf16Dag<half, float>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE8) {
        status = Tiling4ReduceOp<ReduceMin::ReduceMinDag<int64_t>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<ReduceMin::ReduceMinDag<float>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<ReduceMin::ReduceMinDag<half>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == 1) {
        status = Tiling4ReduceOp<ReduceMin::ReduceMinDag<int8_t>::OpDag>(context, opInput, key);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "x", Ops::Base::ToString(opInput.inputDtype).c_str(),
            "int8, uint8, bfloat16, float16, float, int32 or int64"),
        return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus Tiling4ReduceMin(gert::TilingContext* context)
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
        OP_LOGE(context->GetNodeName(), "DoTiling Failed for ReduceMin"), return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(
        context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu", key.patternID,
        key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReduceMin(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReduceMin).Tiling(Tiling4ReduceMin).TilingParse<ReduceOpCompileInfo>(TilingPrepare4ReduceMin);
} // namespace optiling