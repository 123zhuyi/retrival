/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bias_tiling_arch35.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/arch35/bias_tiling_key.h"
#include <graph/utils/type_utils.h>
#include <vector>
#include <cstring>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

namespace optiling {
namespace {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_BIAS_INDEX = 1;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t ATTR_AXIS_INDEX = 0;
constexpr size_t ATTR_NUM_AXES_INDEX = 1;
constexpr size_t ATTR_BIAS_FROM_BLOB_INDEX = 2;
constexpr int64_t DEFAULT_AXIS = 1;
constexpr int64_t DEFAULT_NUM_AXES = 1;
constexpr bool DEFAULT_BIAS_FROM_BLOB = true;
constexpr uint64_t WORKSPACE_SIZE = 16777216;

constexpr uint32_t SCH_MODE_FP32 = BIAS_TPL_SCH_MODE_FLOAT32;
constexpr uint32_t SCH_MODE_FP16 = BIAS_TPL_SCH_MODE_FLOAT16;
constexpr uint32_t SCH_MODE_BF16 = BIAS_TPL_SCH_MODE_BFLOAT16;

int64_t GetAttrInt(const gert::RuntimeAttrs* attrs, size_t index, int64_t defaultValue)
{
    if (attrs == nullptr || attrs->GetAttrNum() <= index) {
        return defaultValue;
    }
    const int64_t* value = attrs->GetAttrPointer<int64_t>(index);
    return value == nullptr ? defaultValue : *value;
}

bool GetAttrBool(const gert::RuntimeAttrs* attrs, size_t index, bool defaultValue)
{
    if (attrs == nullptr || attrs->GetAttrNum() <= index) {
        return defaultValue;
    }
    const bool* value = attrs->GetAttrPointer<bool>(index);
    return value == nullptr ? defaultValue : *value;
}

bool IsSupportedDtype(const ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16;
}

bool IsRightAlignedEquivalent(const gert::Shape& logicalShape, const gert::Shape& storageShape)
{
    if (storageShape.GetDimNum() > logicalShape.GetDimNum()) {
        return false;
    }
    const size_t rankDiff = logicalShape.GetDimNum() - storageShape.GetDimNum();
    for (size_t i = 0; i < logicalShape.GetDimNum(); ++i) {
        const int64_t storageDim = i < rankDiff ? 1 : storageShape.GetDim(i - rankDiff);
        if (logicalShape.GetDim(i) != storageDim) {
            return false;
        }
    }
    return true;
}

std::map<uint64_t, Ops::Base::BroadcastComputeParams> GetComputeMap(ge::DataType dtype)
{
    Ops::Base::BroadcastComputeParams params;
    params.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
    params.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS1_SIZE);
    params.extraSize = {0, 0};
    switch (dtype) {
        case DT_FLOAT:
            params.bufferDivisor = {256, 256};
            return {{1, params}};
        case DT_FLOAT16:
            params.bufferDivisor = {128, 128};
            return {{1, params}};
        case DT_BF16:
            params.bufferDivisor = {128, 128};
            return {{1, params}};
        default:
            return {};
    }
}

bool InferBiasShape(gert::TilingContext* context, std::vector<gert::Shape>& inputShapes)
{
    auto xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    auto biasStorageShape = context->GetInputShape(INPUT_BIAS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasStorageShape);

    const gert::Shape& xShape = Ops::Base::EnsureNotScalar(xStorageShape->GetStorageShape());
    const gert::Shape& biasShape = Ops::Base::EnsureNotScalar(biasStorageShape->GetStorageShape());
    gert::Shape broadcastBiasShape;

    const auto attrs = context->GetAttrs();
    const int64_t xDimNum = static_cast<int64_t>(xShape.GetDimNum());
    const int64_t biasDimNum = static_cast<int64_t>(biasShape.GetDimNum());
    int64_t axis = GetAttrInt(attrs, ATTR_AXIS_INDEX, DEFAULT_AXIS);
    const int64_t numAxes = GetAttrInt(attrs, ATTR_NUM_AXES_INDEX, DEFAULT_NUM_AXES);
    const bool biasFromBlob = GetAttrBool(attrs, ATTR_BIAS_FROM_BLOB_INDEX, DEFAULT_BIAS_FROM_BLOB);

    OP_CHECK_IF(numAxes < -1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "num_axes", std::to_string(numAxes).c_str(), ">= -1"),
                return false);
    OP_CHECK_IF(xDimNum < 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDimNum).c_str(), ">= 1"),
                return false);
    OP_CHECK_IF(axis >= xDimNum || axis < -xDimNum,
                OP_LOGE_FOR_INVALID_VALUE(
                    context->GetNodeName(), "axis", std::to_string(axis).c_str(),
                    ("in [" + std::to_string(-xDimNum) + ", " + std::to_string(xDimNum - 1) + "]").c_str()),
                return false);
    if (axis < 0) {
        axis += xDimNum;
    }

    if (biasFromBlob) {
        OP_CHECK_IF(numAxes == -1 && biasDimNum != xDimNum - axis,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bias", std::to_string(biasDimNum).c_str(),
                                                 std::to_string(xDimNum - axis).c_str()),
                    return false);
        OP_CHECK_IF(
            numAxes == 0 && biasDimNum != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bias", std::to_string(biasDimNum).c_str(), "1"),
            return false);
        OP_CHECK_IF(numAxes > 0 && axis + numAxes > xDimNum,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "num_axes", std::to_string(numAxes).c_str(),
                                              "axis + num_axes <= rank(x)"),
                    return false);
        OP_CHECK_IF(numAxes > 0 && biasDimNum != numAxes,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bias", std::to_string(biasDimNum).c_str(),
                                                 std::to_string(numAxes).c_str()),
                    return false);

        if (numAxes == -1) {
            for (int64_t i = 0; i < axis; ++i)
                broadcastBiasShape.AppendDim(1);
            for (size_t i = 0; i < biasShape.GetDimNum(); ++i)
                broadcastBiasShape.AppendDim(biasShape.GetDim(i));
        } else if (numAxes == 0) {
            for (int64_t i = 0; i < xDimNum; ++i)
                broadcastBiasShape.AppendDim(1);
        } else {
            for (int64_t i = 0; i < axis; ++i)
                broadcastBiasShape.AppendDim(1);
            for (size_t i = 0; i < biasShape.GetDimNum(); ++i)
                broadcastBiasShape.AppendDim(biasShape.GetDim(i));
            for (int64_t i = 0; i < xDimNum - numAxes - axis; ++i)
                broadcastBiasShape.AppendDim(1);
        }
    } else {
        OP_CHECK_IF(biasDimNum != 1 && axis + biasDimNum > xDimNum,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bias", std::to_string(biasDimNum).c_str(),
                                                 "axis + rank(bias) <= rank(x)"),
                    return false);
        if (biasDimNum == 1 && biasShape.GetDim(0) == 1) {
            for (int64_t i = 0; i < xDimNum; ++i)
                broadcastBiasShape.AppendDim(1);
        } else {
            for (int64_t i = 0; i < axis; ++i)
                broadcastBiasShape.AppendDim(1);
            for (size_t i = 0; i < biasShape.GetDimNum(); ++i)
                broadcastBiasShape.AppendDim(biasShape.GetDim(i));
            for (int64_t i = 0; i < xDimNum - biasDimNum - axis; ++i)
                broadcastBiasShape.AppendDim(1);
        }
    }

    OP_CHECK_IF(broadcastBiasShape.GetDimNum() != xShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bias",
                                             std::to_string(broadcastBiasShape.GetDimNum()).c_str(),
                                             std::to_string(xShape.GetDimNum()).c_str()),
                return false);
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        const int64_t biasDim = broadcastBiasShape.GetDim(i);
        const int64_t xDim = xShape.GetDim(i);
        OP_CHECK_IF(
            biasDim != 1 && biasDim != xDim,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "bias", ToString(broadcastBiasShape).c_str(),
                                                  "Each broadcast bias dimension must be 1 or equal to x dimension."),
            return false);
    }

    if (!IsRightAlignedEquivalent(broadcastBiasShape, biasShape)) {
        inputShapes.push_back(xShape);
        inputShapes.push_back(broadcastBiasShape);
    }
    return true;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = static_cast<const BiasCompileInfo*>(context->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = ubSizePlatform;
    }
    return ge::GRAPH_SUCCESS;
}

ge::DataType GetShapeAttrsInfo(gert::TilingContext* context, uint32_t& schMode)
{
    auto xDesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "x desc is null"), return ge::DT_UNDEFINED);
    auto biasDesc = context->GetInputDesc(INPUT_BIAS_INDEX);
    OP_CHECK_IF(biasDesc == nullptr, OP_LOGE(context->GetNodeName(), "bias desc is null"), return ge::DT_UNDEFINED);
    auto outputDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_IF(outputDesc == nullptr, OP_LOGE(context->GetNodeName(), "output desc is null"), return ge::DT_UNDEFINED);

    const ge::DataType xDType = xDesc->GetDataType();
    const ge::DataType biasDType = biasDesc->GetDataType();
    const ge::DataType outputDType = outputDesc->GetDataType();
    OP_CHECK_IF(xDType != biasDType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x and bias",
                                                       (ge::TypeUtils::DataTypeToSerialString(xDType) + " and " +
                                                        ge::TypeUtils::DataTypeToSerialString(biasDType))
                                                           .c_str(),
                                                       "The dtypes of x and bias must be the same."),
                return ge::DT_UNDEFINED);
    OP_CHECK_IF(xDType != outputDType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x and y",
                                                       (ge::TypeUtils::DataTypeToSerialString(xDType) + " and " +
                                                        ge::TypeUtils::DataTypeToSerialString(outputDType))
                                                           .c_str(),
                                                       "The dtypes of x and y must be the same."),
                return ge::DT_UNDEFINED);
    OP_CHECK_IF(
        !IsSupportedDtype(xDType),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDType).c_str(),
                                  "float16, float32, bfloat16"),
        return ge::DT_UNDEFINED);

    schMode = (xDType == DT_FLOAT)   ? SCH_MODE_FP32 :
              (xDType == DT_FLOAT16) ? SCH_MODE_FP16 :
              (xDType == DT_BF16)    ? SCH_MODE_BF16 :
                                       0;
    OP_CHECK_IF(
        schMode == 0 && xDType != DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDType).c_str(),
                                  "float16, float32, bfloat16"),
        return ge::DT_UNDEFINED);
    return xDType;
}
} // namespace

ge::graphStatus TilingForBias(gert::TilingContext* context)
{
    OP_LOGD("TilingForBias", "Enter TilingForBias");

    auto* tiling = context->GetTilingData<BiasTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BiasTilingData), 0, sizeof(BiasTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data failed"), return ge::GRAPH_FAILED);

    uint32_t schMode = 0;
    ge::DataType xDType = GetShapeAttrsInfo(context, schMode);
    OP_CHECK_IF(xDType == ge::DT_UNDEFINED, OP_LOGE(context, "GetShapeAttrsInfo failed"), return ge::GRAPH_FAILED);

    int64_t coreNum = 0;
    int64_t ubSize = 0;
    OP_CHECK_IF(GetPlatformInfo(context, coreNum, ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;

    std::vector<gert::Shape> inputShapes;
    OP_CHECK_IF(!InferBiasShape(context, inputShapes),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "bias", "invalid", "valid broadcast shape"),
                return ge::GRAPH_FAILED);

    auto xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::Shape& xShape = Ops::Base::EnsureNotScalar(xStorageShape->GetStorageShape());
    int64_t elemNum = xShape.GetShapeSize();

    if (elemNum == 0) {
        tiling->blockFormer = 0;
        tiling->blockTail = 0;
        tiling->ubFormer = 0;
        tiling->ubTail = 0;
        tiling->ubOuter = 0;
        tiling->elemNum = 0;
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, schMode);
        return ge::GRAPH_SUCCESS;
    }

    Ops::Base::BroadcastTilingParams broadcastTilingParams;
    for (uint64_t i = 0; i < context->GetComputeNodeInputNum(); i++) {
        auto shape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        broadcastTilingParams.inShape.push_back(Ops::Base::EnsureNotScalar(shape->GetStorageShape()));
    }
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    broadcastTilingParams.outShape = Ops::Base::EnsureNotScalar(outputShape->GetStorageShape());
    broadcastTilingParams.computeMap = GetComputeMap(xDType);
    broadcastTilingParams.coreNum = coreNum;
    broadcastTilingParams.ubSize = ubSize;

    if (!inputShapes.empty()) {
        broadcastTilingParams.inShape.clear();
        for (auto& s : inputShapes) {
            broadcastTilingParams.inShape.push_back(s);
        }
    }

    Ops::Base::BroadcastTilingData broadcastTilingData;
    ge::graphStatus status = BroadcastTiling(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "broadcast tiling failed.");
        return ge::GRAPH_FAILED;
    }

    tiling->blockFormer = broadcastTilingData.blockFormer;
    tiling->ubFormer = broadcastTilingData.ubFormer;
    tiling->ubOuter = broadcastTilingData.ubOuter;
    tiling->ubTail = broadcastTilingData.ubTail;
    tiling->blockTail = broadcastTilingData.blockTail;
    tiling->shapeLen = broadcastTilingData.shapeLen;
    tiling->ubSplitAxis = broadcastTilingData.ubSplitAxis;
    tiling->dimProductBeforeUbInner = broadcastTilingData.dimProductBeforeUbInner;
    tiling->elemNum = broadcastTilingData.elemNum;

    for (int64_t i = 0; i < BIAS_MAX_DIM_SIZE; ++i) {
        tiling->input0Dims[i] = broadcastTilingData.dims[0].size() > static_cast<size_t>(i) ?
                                    broadcastTilingData.dims[0][i] :
                                    0;
        tiling->input1Dims[i] = broadcastTilingData.dims[1].size() > static_cast<size_t>(i) ?
                                    broadcastTilingData.dims[1][i] :
                                    0;
        tiling->outputDims[i] = broadcastTilingData.dims[2].size() > static_cast<size_t>(i) ?
                                    broadcastTilingData.dims[2][i] :
                                    0;
        tiling->input0Strides[i] = broadcastTilingData.strides[0].size() > static_cast<size_t>(i) ?
                                       broadcastTilingData.strides[0][i] :
                                       0;
        tiling->input1Strides[i] = broadcastTilingData.strides[1].size() > static_cast<size_t>(i) ?
                                       broadcastTilingData.strides[1][i] :
                                       0;
        tiling->outputStrides[i] = broadcastTilingData.strides[2].size() > static_cast<size_t>(i) ?
                                       broadcastTilingData.strides[2][i] :
                                       0;
    }

    context->SetBlockDim(broadcastTilingData.blockNum);
    ASCENDC_TPL_SEL_PARAM(context, schMode);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForBias(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<BiasCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    OP_CHECK_IF(compileInfoPtr->coreNum == 0 || compileInfoPtr->ubSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "totalCoreNum,ubSize",
                    std::to_string(compileInfoPtr->coreNum) + ", " + std::to_string(compileInfoPtr->ubSize),
                    "The values of totalCoreNum and ubSize must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Bias).Tiling(TilingForBias).TilingParse<BiasCompileInfo>(TilingParseForBias);
} // namespace optiling
