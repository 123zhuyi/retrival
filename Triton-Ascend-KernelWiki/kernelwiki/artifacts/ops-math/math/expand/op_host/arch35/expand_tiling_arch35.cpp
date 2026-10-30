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
 * \file expand_tiling_arch35.cpp
 * \brief calc tiling for expand
 */
#include "expand_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "op_host/util/const_util.h"
#include "op_host/tiling_base_util.h"
#include "util/platform_util.h"
#include <sstream>

namespace optiling {
constexpr size_t MAX_DIM_NUM = 0x8;
constexpr size_t BRCTO_MAX_DIM_NUM = 0x8;

template <typename T>
inline std::string ConcatString(const T& arg)
{
    std::ostringstream oss;
    oss << arg;
    return oss.str();
}

template <typename T, typename... Ts>
static std::string ConcatString(const T& arg, const Ts&... arg_left)
{
    std::ostringstream oss;
    oss << arg;
    oss << ConcatString(arg_left...);
    return oss.str();
}

void AdjustShapesToSameDimNum(gert::Shape& inShape, size_t outDimNum)
{
    auto inDimNum = inShape.GetDimNum();
    if (inDimNum >= outDimNum) {
        return;
    }

    gert::Shape newShape;
    size_t gapSize = outDimNum - inDimNum;
    for (size_t i = 0; i < gapSize; i++) {
        newShape.AppendDim(1);
    }
    for (size_t j = 0; j < inDimNum; j++) {
        newShape.AppendDim(inShape[j]);
    }
    inShape = newShape;
}

ge::graphStatus GetABFlag(const gert::TilingContext* context, const gert::Shape& inShape, const gert::Shape& outShape,
                          std::array<bool, MAX_DIM_NUM>& abInfo)
{
    size_t inDimNum;
    if (brcto::CheckSameDimNum(context, inShape, outShape, inDimNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    for (size_t idx = 0; idx < inDimNum; idx++) {
        abInfo[idx] = (inShape[idx] != outShape[idx]);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MergeAxis(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    size_t dimNum;
    if (brcto::CheckSameDimNum(context, inShape, outShape, dimNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (dimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }

    std::array<bool, MAX_DIM_NUM> abInfo{};
    if (GetABFlag(context, inShape, outShape, abInfo) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "unknown";
        std::string reasonMsg = "Failed to get axes info.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    size_t mIdx = 0;
    for (size_t oIdx = 1; oIdx < dimNum; oIdx++) {
        if (abInfo[oIdx] == abInfo[mIdx]) {
            inShape[mIdx] *= inShape[oIdx];
            outShape[mIdx] *= outShape[oIdx];
        } else {
            mIdx += 1;
            inShape[mIdx] = inShape[oIdx];
            outShape[mIdx] = outShape[oIdx];
            abInfo[mIdx] = abInfo[oIdx];
        }
    }
    inShape.SetDimNum(mIdx + 1);
    outShape.SetDimNum(mIdx + 1);

    return ge::GRAPH_SUCCESS;
}

// Read outShape from shape tensor (index 1)
static ge::graphStatus ReadOutShapeFromTensor(const gert::TilingContext* context, gert::Shape& outShape)
{
    auto shapeTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeTensor);
    auto shapeSize = static_cast<size_t>(shapeTensor->GetShapeSize());
    outShape.SetDimNum(shapeSize);

    ge::DataType dataType = shapeTensor->GetDataType();
    if ((dataType != ge::DT_INT32) && (dataType != ge::DT_INT64)) {
        std::string dtypeMsg = Ops::Base::ToString(dataType);
        std::string reasonMsg = "shape's dtype must be in (int32, int64).";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "shape", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (dataType == ge::DT_INT32) {
        const int32_t* values = shapeTensor->GetData<int32_t>();
        for (size_t i = 0; i < shapeSize; i++) {
            outShape.SetDim(i, static_cast<int64_t>(values[i]));
        }
    } else {
        const int64_t* values = shapeTensor->GetData<int64_t>();
        for (size_t i = 0; i < shapeSize; i++) {
            outShape.SetDim(i, values[i]);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyBroadcastRules(const gert::TilingContext* context, gert::Shape& inShape,
                                           gert::Shape& outShape)
{
    size_t outDimNum = outShape.GetDimNum();

    for (size_t i = 0; i < outDimNum; i++) {
        int64_t xDim = inShape.GetDim(i);
        int64_t outDim = outShape.GetDim(i);
        if (outDim == -1) {
            outShape.SetDim(i, xDim);
            continue;
        }
        if (outDim == 1 && xDim != 1) {
            outShape.SetDim(i, xDim);
            continue;
        }
        if (xDim != 1 && xDim != outDim) {
            std::string shapeMsg = "unknown";
            std::string reasonMsg = "x dimension " + std::to_string(xDim) + " at axis " + std::to_string(i) +
                                    " cannot be broadcast to " + std::to_string(outDim) + ".";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(),
                                                   reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsEmptyTensor(const gert::TilingContext* context)
{
    auto xStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    gert::Shape xShape = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());

    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    gert::Shape yShape = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());

    auto xDimNum = xShape.GetDimNum();
    auto yDimNum = yShape.GetDimNum();

    for (size_t i = 0; i < xDimNum; i++) {
        if (xShape[i] == 0) {
            return ge::GRAPH_FAILED;
        }
    }

    for (size_t i = 0; i < yDimNum; i++) {
        if (yShape[i] == 0) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ProcessBroadcastShapes(const gert::TilingContext* context, gert::Shape& inShape,
                                              gert::Shape& outShape)
{
    auto outDimNum = outShape.GetDimNum();
    AdjustShapesToSameDimNum(inShape, outDimNum);
    if (ApplyBroadcastRules(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "unknown";
        std::string reasonMsg = "Failed to apply broadcast rules.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "%s",
            ConcatString("input0 and input1 infer output, output shape is ", Ops::Base::ToString(outShape).c_str())
                .c_str());
    if (brcto::DeleteOneSizeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "unknown";
        std::string reasonMsg = "Failed to delete one size axes.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (MergeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "unknown";
        std::string reasonMsg = "Failed to merge axes.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "%s",
            ConcatString("input0 shape MergeAxis result is ", Ops::Base::ToString(inShape).c_str(),
                         ", output shape MergeAxis result is ", Ops::Base::ToString(outShape).c_str())
                .c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetShapeInfo(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    auto xStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    inShape = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());

    if (IsEmptyTensor(context) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "empty";
        std::string reasonMsg = "Do not support x or y is empty tensor.";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x or y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (ReadOutShapeFromTensor(context, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "unknown";
        std::string reasonMsg = "Failed to read outShape from tensor.";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "shape", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "%s",
            ConcatString("input0 shape is ", Ops::Base::ToString(inShape).c_str(), ", input1 shape is ",
                         Ops::Base::ToString(outShape).c_str())
                .c_str());
    auto outDimNum = outShape.GetDimNum();
    if (inShape.GetDimNum() > outDimNum) {
        std::string dimMsg = std::to_string(inShape.GetDimNum()) + " and " + std::to_string(outDimNum);
        std::string reasonMsg = "The input0 shape should not have more dimensions than input1 shape.";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x and y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (outDimNum > BRCTO_MAX_DIM_NUM) {
        std::string dimMsg = std::to_string(outDimNum);
        std::string reasonMsg = "The output dim num should not be greater than " + std::to_string(BRCTO_MAX_DIM_NUM);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ProcessBroadcastShapes(context, inShape, outShape);
}

ge::graphStatus Tiling4ExpandAscendC(gert::TilingContext* context, const gert::Shape* inShapePtr,
                                     const gert::Shape* outShapePtr)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, inShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    brcto::BroadcastToTilingAscendC brcToTiling(context, inShapePtr, outShapePtr);

    if (brcToTiling.GetHardwareInfo<ExpandCompileInfo>() != ge::GRAPH_SUCCESS) {
        std::string valueMsg = "unknown";
        std::string reasonMsg = "BroadcastToTilingAscendC failed to get hardware info.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "hardwareInfo", valueMsg.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return brcToTiling.DoTiling();
}

static ge::graphStatus Tiling4Expand(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const ExpandCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    gert::Shape inShape;
    gert::Shape outShape;

    if (GetShapeInfo(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "unknown";
        std::string reasonMsg = "Failed to get input or output shape.";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x or y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return Tiling4ExpandAscendC(context, &inShape, &outShape);
}

static ge::graphStatus TilingPrepare4Expand(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4Expand.");

    auto compileInfo = context->GetCompiledInfo<ExpandCompileInfo>();
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

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4Expand.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Expand op.
IMPL_OP_OPTILING(Expand).Tiling(Tiling4Expand).TilingParse<ExpandCompileInfo>(TilingPrepare4Expand);
} // namespace optiling
