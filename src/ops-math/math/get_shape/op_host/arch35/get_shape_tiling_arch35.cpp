/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "math/get_shape/op_kernel/arch35/get_shape_tiling_data.h"
#include "get_shape_tiling_arch35.h"

namespace optiling {

static ge::graphStatus GetShapeTilingFunc(gert::TilingContext* context)
{
    context->SetBlockDim(1);
    context->SetTilingKey(0);

    auto computeNodeInfo = context->GetComputeNodeInfo();
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(0);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();

    if (inputNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", "0",
                                              "The value of x must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    uint32_t totalDimNum = 0;
    for (uint32_t i = 0; i < inputNum; ++i) {
        auto xShape = context->GetDynamicInputShape(0, i);
        if (xShape == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(i).c_str(),
                                                  "The input shape of x must not be nullptr");
            return ge::GRAPH_FAILED;
        }
        if (Ops::Base::IsUnknownRank(xShape->GetStorageShape())) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", "unknown rank",
                                                  "x cannot be an unknown rank tensor");
            return ge::GRAPH_FAILED;
        }
        auto dimNum = xShape->GetStorageShape().GetDimNum();
        if (static_cast<uint32_t>(dimNum) > static_cast<uint32_t>(GetShapeConst::MAX_DIM_PER_TENSOR)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(dimNum).c_str(), "8");
            return ge::GRAPH_FAILED;
        }
        totalDimNum += dimNum;
    }

    OP_LOGD(context->GetNodeName(), "[GetShape] tiling totalDimNum=%u, inputNum=%u", totalDimNum, inputNum);
    if (totalDimNum == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "y", "0", "greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (totalDimNum > static_cast<uint32_t>(GetShapeConst::MAX_TOTAL_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "y", std::to_string(totalDimNum).c_str(), "128");
        return ge::GRAPH_FAILED;
    }

    auto tilingData = context->GetTilingData<GetShapeTilingData>();
    tilingData->inputNum = static_cast<int32_t>(inputNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGetShape([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GetShape).Tiling(GetShapeTilingFunc).TilingParse<GetShapeCompileInfo>(TilingParseForGetShape);

} // namespace optiling
