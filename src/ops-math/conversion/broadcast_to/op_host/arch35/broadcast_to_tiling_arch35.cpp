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
 * \brief BroadcastTo tiling: 通用 tiling + 单轴特化 tiling
 */
#include "broadcast_to_tiling_arch35.h"
#include "broadcast_to_tiling_base.h"
#include "../../op_kernel/arch35/broadcast_to_with_single_axis_tiling_data.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include <string>
#include <algorithm>

using namespace ge;

namespace optiling {
constexpr size_t INPUT_INDEX_SHAPE = 1;

constexpr int64_t TILING_MODE_SINGLE_AXIS = 11006;
constexpr int64_t TILING_MODE_SINGLE_AXIS_BRC = 11007;
constexpr uint32_t UB_BLOCK_BYTES = 128;
constexpr uint32_t MIN_UB_BUFFER_BYTES = 8 * 1024;
constexpr uint32_t BUFFER_NUM_A = 2; // A轴: double buffer (ubPing + ubPong)
constexpr uint32_t BUFFER_NUM_B = 1; // B轴: 单buffer常驻UB, 无需double buffer

static ge::graphStatus Tiling4SingleAxis(gert::TilingContext* context, const gert::Shape& inShape,
                                         const gert::Shape& outShape)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizeU = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeU);

    if (coreNum == 0 || ubSizeU == 0) {
        std::string reasonMsg = "Invalid hardware info: coreNum or ubSize is not positive.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "hardwareInfo", "unknown", reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int32_t dtypeSize = GetSizeByDataType(inputDesc->GetDataType());
    if (dtypeSize <= 0) {
        std::string reasonMsg = "Unsupported data type for single axis broadcast.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype", "unknown", reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t inDim = inShape.GetDim(0);
    int64_t outDim = outShape.GetDim(0);
    uint64_t totalOutElems = static_cast<uint64_t>(outDim);
    bool isBrc = (inDim != outDim);

    // 1. 先切多核: 负载均衡, 初步算每核元素数
    uint64_t blockFactor = Ops::Base::CeilDiv(totalOutElems, static_cast<uint64_t>(coreNum));
    uint64_t usedCoreNum = std::min(static_cast<uint64_t>(coreNum), totalOutElems);
    uint64_t mainCoreNum = totalOutElems - (blockFactor - 1) * usedCoreNum;

    // 2. 再切UB: 按可用UB最大值切分, 128B对齐, tileSize至少8KB
    //    BRC单buffer常驻, 可用全部UB; A轴double buffer对半分
    uint32_t bufferNum = isBrc ? BUFFER_NUM_B : BUFFER_NUM_A;
    uint32_t bufferSize = static_cast<uint32_t>(ubSizeU / bufferNum);
    bufferSize = bufferSize / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
    uint32_t bufferSizeElements = bufferSize / static_cast<uint32_t>(dtypeSize);
    uint32_t minUbElements = MIN_UB_BUFFER_BYTES / static_cast<uint32_t>(dtypeSize);
    uint32_t tileSize = std::max(std::min(static_cast<uint32_t>(blockFactor), bufferSizeElements), minUbElements);
    // 对齐到128B(UB_BLOCK_BYTES): Duplicate(tensor版, arch35)底层是UB→UB DMA(MTE), 要求128B对齐
    uint32_t alignElements = UB_BLOCK_BYTES / static_cast<uint32_t>(dtypeSize);
    tileSize = tileSize / alignElements * alignElements;
    if (tileSize == 0) {
        tileSize = alignElements;
    }

    // 3. 用tileSize重新计算总tile数和核数, 负载均衡
    uint64_t totalTiles = Ops::Base::CeilDiv(totalOutElems, static_cast<uint64_t>(tileSize));
    usedCoreNum = std::min(static_cast<uint64_t>(coreNum), totalTiles);
    blockFactor = Ops::Base::CeilDiv(totalTiles, usedCoreNum);
    mainCoreNum = totalTiles - (blockFactor - 1) * usedCoreNum;

    uint64_t tilingKey = isBrc ? TILING_MODE_SINGLE_AXIS_BRC : TILING_MODE_SINGLE_AXIS;

    auto tilingData = context->GetTilingData<BrcSA::SingleAxisBrcTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    tilingData->shapeSize = totalOutElems;
    tilingData->tileSize = tileSize;
    tilingData->blockNum = static_cast<uint32_t>(usedCoreNum);
    tilingData->blockFactor = blockFactor;

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    OP_LOGI(context->GetNodeName(),
            "SingleAxis tiling: key=%lu, shapeSize=%lu, blockFactor=%lu, mainCoreNum=%lu, usedCoreNum=%lu, "
            "bufferSize=%u, tileSize=%u",
            tilingKey, totalOutElems, blockFactor, mainCoreNum, usedCoreNum, bufferSize, tileSize);

    return ge::GRAPH_SUCCESS;
}

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

    if (brcto::IsSingleAxisBrcAfterMerge(outShape)) {
        OP_LOGI(context->GetNodeName(), "Enter single axis specialization tiling after merge.");
        return Tiling4SingleAxis(context, inShape, outShape);
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

IMPL_OP_OPTILING(BroadcastTo).Tiling(Tiling4BroadcastTo).TilingParse<BroadcastToCompileInfo>(TilingPrepare4BroadcastTo);
} // namespace optiling
