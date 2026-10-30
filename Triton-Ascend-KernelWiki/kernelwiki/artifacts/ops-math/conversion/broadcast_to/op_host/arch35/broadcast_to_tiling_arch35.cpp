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
 * \file broadcast_to_tiling_arch35.cpp
 * \brief
 */
#include "broadcast_to_tiling_arch35.h"
#include "broadcast_to_tiling_base.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"

using namespace ge;

namespace optiling {
constexpr size_t INPUT_INDEX_SHAPE = 1;

static ge::graphStatus Tiling4BroadcastTo(gert::TilingContext* context)
{
    auto compile_info = context->GetCompileInfo<BroadcastToCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    gert::Shape inShape;
    gert::Shape outShape;
    if (brcto::GetShapeInfo(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "unknown";
        std::string reasonMsg = "Failed to get input or output shape.";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x or y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return Tiling4BroadcastToAscendC(context, &inShape, &outShape);
}

static ge::graphStatus TilingPrepare4BrcToAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4BrcToAscendC.");

    auto compileInfo = context->GetCompiledInfo<BroadcastToCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum <= 0) {
        std::string valueMsg = std::to_string(compileInfo->coreNum);
        std::string reasonMsg = "The core num must be positive.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    if (compileInfo->ubSize <= 0) {
        std::string valueMsg = std::to_string(compileInfo->ubSize);
        std::string reasonMsg = "Failed to get ub size, ub size must be positive.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    compileInfo->clSize = Ops::Base::GetCacheLineSize(context);
    if (compileInfo->clSize <= 0) {
        std::string valueMsg = std::to_string(compileInfo->clSize);
        std::string reasonMsg = "Failed to get cache line size, cache line size must be positive.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "clSize", valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    if (compileInfo->blockSize <= 0) {
        std::string valueMsg = std::to_string(compileInfo->blockSize);
        std::string reasonMsg = "Failed to get block size, block size must be positive.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockSize", valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
    if (compileInfo->vRegSize <= 0) {
        std::string valueMsg = std::to_string(compileInfo->vRegSize);
        std::string reasonMsg = "Failed to get vReg size, vReg size must be positive.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "vRegSize", valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4BrcToAscendC.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4BroadcastTo(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<BroadcastToCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD("TilingPrepare4BroadcastTo", "AscendC TilingPrepare4BroadcastTo success.");
    return TilingPrepare4BrcToAscendC(context);
}

// register tiling interface of the BroadcastTo op.
IMPL_OP_OPTILING(BroadcastTo).Tiling(Tiling4BroadcastTo).TilingParse<BroadcastToCompileInfo>(TilingPrepare4BroadcastTo);
} // namespace optiling
