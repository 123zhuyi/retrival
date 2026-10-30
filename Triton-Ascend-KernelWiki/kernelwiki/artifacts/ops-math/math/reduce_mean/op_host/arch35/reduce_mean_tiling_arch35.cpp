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
 * \file reduce_mean_tiling_arch35.cpp
 * \brief
 */

#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "op_host/tiling_base_util.h"
#include "util/platform_util.h"
#include "math/reduce_mean/op_kernel/arch35/reduce_mean_dag.h"
#include "math/reduce_mean/op_kernel/arch35/reduce_mean_tiling_key.h"

using namespace Ops::Base;

namespace optiling {
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
static constexpr int32_t BATCH_INVARIANT_LEVEL = 3;

static ge::graphStatus DoTilingAscendC(gert::TilingContext* context, const ReduceOpCompileInfo* compileInfo,
                                       ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;

    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<ReduceMean::ReduceMeanDag<float, float>::OpDag>(context, opInput, key, compileInfo);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<ReduceMean::ReduceMeanDag<half, float>::OpDag>(context, opInput, key, compileInfo);
    }

    OP_CHECK_IF((status == ge::GRAPH_FAILED),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(opInput.inputDtype).c_str(),
                                          "bfloat16, float16 or float"),
                return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus Tiling4ReduceMean(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    ReduceOpInputParam opInput;
    ReduceTilingKey key;
    // 获取确定性级别，如果为3开启batch一致性
    if (context->GetDeterministicLevel() == BATCH_INVARIANT_LEVEL) {
        key.batchInvariant = 1;
    }
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(context, opInput, 0, 1, 0, key) == ge::GRAPH_FAILED),
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

    OP_CHECK_IF((DoTilingAscendC(context, compile_info, opInput, key) == ge::GRAPH_FAILED),
                OP_LOGE(context->GetNodeName(), "DoTiling Failed for ReduceMean"), return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(context->GetNodeName(),
            "patternID:%u, loopARCount:%u, loopInnerARCount:%u, isContiguous:%d, batchInvariant:%d, Tiling Key is:%lu",
            key.patternID, key.loopARCount, key.loopInnerARCount, key.isContiguous ? 1 : 0, key.batchInvariant ? 1 : 0,
            tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReduceMean(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ReduceMean op.
IMPL_OP_OPTILING(ReduceMean).Tiling(Tiling4ReduceMean).TilingParse<ReduceOpCompileInfo>(TilingPrepare4ReduceMean);
} // namespace optiling
