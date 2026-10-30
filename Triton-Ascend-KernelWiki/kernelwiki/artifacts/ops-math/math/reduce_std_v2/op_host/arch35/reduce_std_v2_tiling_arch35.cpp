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
 * \file reduce_std_v2_tiling_arch35.cpp
 * \brief
 */

#include "math/reduce_var/op_host/arch35/reduce_var_tiling.h"
#include "log/log.h"
#include "math/reduce_std_v2/op_kernel/arch35/reduce_std_v2_tiling_key.h"

namespace optiling {
static ge::graphStatus Tiling4ReduceStdV2(gert::TilingContext* context) {
    auto compileInfo = reinterpret_cast<const ReduceVarCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    Ops::Base::ReduceTilingKey key;
    ReduceVarTilingData tilingData;
    Ops::Base::ReduceOpTilingData reduceTiling;
    ReduceVarTiling tiling(context, compileInfo, &tilingData, &reduceTiling);
    OP_CHECK_IF((tiling.RunTiling(key) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "RunTiling Failed for ReduceStdV2"),
        return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu",
        key.patternID, key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReduceStdV2(gert::TilingParseContext* context) {
    OP_CHECK_IF((context == nullptr), OP_LOGE(context->GetNodeName(), "context is nil"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReduceStdV2)
    .Tiling(Tiling4ReduceStdV2)
    .TilingParse<ReduceVarCompileInfo>(TilingPrepare4ReduceStdV2);

}