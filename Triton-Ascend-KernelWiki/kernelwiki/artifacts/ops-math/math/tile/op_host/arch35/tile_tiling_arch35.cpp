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
 * \file tile_tiling_arch35.cpp
 * \brief calc tiling for tile
 */
#include "tile_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "op_host/tiling_base_util.h"
#include "util/platform_util.h"

namespace optiling {
constexpr size_t TILE_MAX_DIM_NUM = 0x8;

template <typename T>
static std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static ge::graphStatus CheckTileRule(const gert::TilingContext* context, const gert::Shape& inShape,
    const gert::Shape& outShape)
{
    auto outDimNum = outShape.GetDimNum();
    if (inShape.GetDimNum() != outDimNum) {
        std::string dimMsg = std::to_string(inShape.GetDimNum()) + " and " + std::to_string(outDimNum);
        std::string reasonMsg = "The input and output shape dim num should be equal.";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "x and y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < outDimNum; i++) {
        if (outShape[i] < inShape[i] || outShape[i] % inShape[i] != 0) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static void ConvertTileAxis2BroadcastToAxis(gert::Shape& inShape, gert::Shape& outShape)
{
    gert::Shape newInShape;
    gert::Shape newOutShape;
    auto DimNum = inShape.GetDimNum();
    for (size_t i = 0; i < DimNum; i++) {
        newInShape.AppendDim(1);
        newInShape.AppendDim(inShape[i]);
        newOutShape.AppendDim(outShape[i] / inShape[i]);
        newOutShape.AppendDim(inShape[i]);
    }
    inShape = newInShape;
    outShape = newOutShape;
}

static ge::graphStatus GetShapeInfo(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    auto xStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    inShape = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    outShape = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());

    auto outDimNum = outShape.GetDimNum();
    if (inShape.GetDimNum() > outDimNum) {
        std::string dimMsg = std::to_string(inShape.GetDimNum()) + " and " + std::to_string(outDimNum);
        std::string reasonMsg = "The input shape should not have more dimensions than output shape.";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "x and y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (outDimNum > TILE_MAX_DIM_NUM) {
        std::string dimMsg = std::to_string(outDimNum);
        std::string reasonMsg = "The output dim num should not be greater than " + std::to_string(TILE_MAX_DIM_NUM) + ".";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    brcto::AdjustShapesToSameDimNum(inShape, outDimNum);
    if (inShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0) {
        std::string shapeMsg = Shape2String(inShape) + " and " + Shape2String(outShape);
        std::string reasonMsg = "The input or output shape must not be empty.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (CheckTileRule(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = Shape2String(inShape) + " and " + Shape2String(outShape);
        std::string reasonMsg = "The input and output shapes mismatch the broadcast rule.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    ConvertTileAxis2BroadcastToAxis(inShape, outShape);

    OP_LOGI(context->GetNodeName(), "The broadcastTo input and output after convertion is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    if (brcto::DeleteOneSizeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = Shape2String(inShape) + " and " + Shape2String(outShape);
        std::string reasonMsg = "Failed to delete one size axes.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "The reshaped input and output is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    if (brcto::MergeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = Shape2String(inShape) + " and " + Shape2String(outShape);
        std::string reasonMsg = "Failed to merge axes.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "The merged input and output is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4TileAscendC(gert::TilingContext* context, const gert::Shape* inShapePtr,
                                          const gert::Shape* outShapePtr)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, inShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    brcto::BroadcastToTilingAscendC brcToTiling(context, inShapePtr, outShapePtr);

    if (brcToTiling.GetHardwareInfo<TileCompileInfo>() != ge::GRAPH_SUCCESS) {
        std::string valueMsg = "unknown";
        std::string reasonMsg = "BroadcastToTilingAscendC failed to get hardware info.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "hardwareInfo", valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return brcToTiling.DoTiling();
}

static ge::graphStatus Tiling4Tile(gert::TilingContext* context) {
  auto compile_info = reinterpret_cast<const TileCompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  gert::Shape inShape;
  gert::Shape outShape;

  // convert tile input shape to broadcastTo input, then use broadcastTo template
  if (GetShapeInfo(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
    std::string shapeMsg = "unknown";
    std::string reasonMsg = "Failed to get input or output shape.";
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
        context->GetNodeName(), "x or y", shapeMsg.c_str(), reasonMsg.c_str());
    return ge::GRAPH_FAILED;
  }
  return Tiling4TileAscendC(context, &inShape, &outShape);
  
}

static ge::graphStatus TilingPrepare4Tile(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4Tile.");

  auto compileInfo = context->GetCompiledInfo<TileCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

  compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
  if (compileInfo->coreNum <= 0) {
    std::string valueMsg = std::to_string(compileInfo->coreNum);
    std::string reasonMsg = "The core num must be positive.";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
        context->GetNodeName(), "coreNum", valueMsg.c_str(), reasonMsg.c_str());
    return ge::GRAPH_FAILED;
  }

  uint64_t ubSize = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  compileInfo->ubSize = static_cast<int64_t>(ubSize);
  if (compileInfo->ubSize <= 0) {
    std::string valueMsg = std::to_string(compileInfo->ubSize);
    std::string reasonMsg = "Failed to get ub size, ub size must be positive.";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
        context->GetNodeName(), "ubSize", valueMsg.c_str(), reasonMsg.c_str());
    return ge::GRAPH_FAILED;
  }

  compileInfo->clSize = Ops::Base::GetCacheLineSize(context);
  if (compileInfo->clSize <= 0) {
    std::string valueMsg = std::to_string(compileInfo->clSize);
    std::string reasonMsg = "Failed to get cache line size, cache line size must be positive.";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
        context->GetNodeName(), "clSize", valueMsg.c_str(), reasonMsg.c_str());
    return ge::GRAPH_FAILED;
  }

  compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
  if (compileInfo->blockSize <= 0) {
    std::string valueMsg = std::to_string(compileInfo->blockSize);
    std::string reasonMsg = "Failed to get block size, block size must be positive.";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
        context->GetNodeName(), "blockSize", valueMsg.c_str(), reasonMsg.c_str());
    return ge::GRAPH_FAILED;
  }

  compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
  if (compileInfo->vRegSize <= 0) {
    std::string valueMsg = std::to_string(compileInfo->vRegSize);
    std::string reasonMsg = "Failed to get vReg size, vReg size must be positive.";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
        context->GetNodeName(), "vRegSize", valueMsg.c_str(), reasonMsg.c_str());
    return ge::GRAPH_FAILED;
  }

  OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4Tile.");
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Tile op.
IMPL_OP_OPTILING(Tile).Tiling(Tiling4Tile).TilingParse<TileCompileInfo>(TilingPrepare4Tile);
}  // namespace optiling
