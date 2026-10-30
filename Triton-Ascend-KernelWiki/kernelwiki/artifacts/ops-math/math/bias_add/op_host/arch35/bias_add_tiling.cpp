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
 * \file    bias_add_tiling.cpp
 * \brief   bias_add_tiling source file
 */

#include "bias_add_tiling.h"
#include "bias_add_tiling_arch35.h" // dsl头文件
#include <graph/utils/type_utils.h>
#include <unordered_set>
#include "log/log.h"
#include "math/bias_add/op_kernel/arch35/bias_add_dag.h"
#include "math/bias_add/op_kernel/arch35/bias_add_struct.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

namespace optiling {
std::string NCHW_STR = "NCHW";
std::string NHWC_STR = "NHWC";
std::string NCDHW_STR = "NCDHW";
std::string NDHWC_STR = "NDHWC";

/* C维度的索引 */
constexpr size_t C_INDEX_NCXX = 1; // NCHW, NCDHW

constexpr static uint64_t BIAS_ADD_COMMON_TILING_PRIORITY = 0;

bool CheckDataFormat(std::string attrDataFormat)
{
    std::unordered_set<std::string> checkList = {NCHW_STR, NHWC_STR, NCDHW_STR, NDHWC_STR};
    if (checkList.find(attrDataFormat) == checkList.end()) {
        return false;
    }

    return true;
}

bool InferBiasShape(gert::TilingContext* context, vector<gert::Shape>& inputShapes)
{
    ge::Format xFormat = context->GetInputDesc(0)->GetStorageFormat();
    gert::Shape broadcastBiasShape;

    auto attrs = context->GetAttrs();
    auto attrNum = attrs->GetAttrNum();
    std::string attrDataFormat;
    if (attrNum == 0) {
        attrDataFormat = "NHWC";
    } else {
        attrDataFormat = attrs->GetStr(0); // 0 is for data_format
    }

    auto xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    auto& xShape = EnsureNotScalar(xStorageShape->GetStorageShape());

    auto biasStorageShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasStorageShape);
    auto& biasShape = EnsureNotScalar(biasStorageShape->GetStorageShape());

    const size_t xDimNum = xShape.GetDimNum();
    const size_t biasDimNum = biasShape.GetDimNum();

    OP_CHECK_IF(
        xDimNum < 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context->GetNodeName(), "x", std::to_string(xDimNum).c_str(), "great than or equal to 2"),
        return false);

    OP_CHECK_IF(
        biasDimNum != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bias", std::to_string(biasDimNum).c_str(), "1D"),
        return false);

    OP_CHECK_IF(
        !CheckDataFormat(attrDataFormat),
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "data_format", attrDataFormat.c_str(), "NCHW, NHWC, NCDHW or NDHWC"),
        return false);

    for (size_t i = 0; i < xDimNum; i++) {
        broadcastBiasShape.AppendDim(1);
    }

    switch (xFormat) {
        case ge::FORMAT_ND:
            if ((attrDataFormat == NCHW_STR) || (attrDataFormat == NCDHW_STR)) {
                OP_CHECK_IF(
                    xShape[C_INDEX_NCXX] != biasShape[0],
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "x", ToString(xShape).c_str(),
                        "The C-dimension must be equal to bias length " + std::to_string(biasShape[0]) +
                        " when the format of x is ND and attribute data_format is NCHW or NCDHW,"
                        " where C-dimension is the 1st axis of x in the current format"),
                    return false);

                broadcastBiasShape[C_INDEX_NCXX] = xShape.GetDim(C_INDEX_NCXX);
            } else {
                OP_CHECK_IF(
                    xShape[xDimNum - 1] != biasShape[0],
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "x", ToString(xShape).c_str(),
                        "The C-dimension must be equal to bias length " + std::to_string(biasShape[0]) +
                        " when the format of x is ND and attribute data_format is NHWC or NDHWC,"
                        " where C-dimension is the last axis of x in the current format"),
                    return false);

                broadcastBiasShape[xDimNum - 1] = xShape.GetDim(xDimNum - 1);
            }
            break;
        case ge::FORMAT_NCHW:
        case ge::FORMAT_NCDHW:
            OP_CHECK_IF(
                xShape[C_INDEX_NCXX] != biasShape[0],
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", ToString(xShape).c_str(),
                    "The C-dimension must be equal to bias length " + std::to_string(biasShape[0]) +
                    " when the format of x is NCHW or NCDHW, where C-dimension is the 1st axis of x in the current format"),
                return false);

            broadcastBiasShape[C_INDEX_NCXX] = xShape.GetDim(C_INDEX_NCXX);
            break;
        case ge::FORMAT_NHWC:
        case ge::FORMAT_NDHWC:
            OP_CHECK_IF(
                xShape[xDimNum - 1] != biasShape[0],
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", ToString(xShape).c_str(),
                    "The C-dimension must be equal to bias length " + std::to_string(biasShape[0]) +
                    " when the format of x is NHWC or NDHWC, where C-dimension is the last axis of x in the current format"),
                return false);

            broadcastBiasShape[xDimNum - 1] = xShape.GetDim(xDimNum - 1);
            break;
        default:
            OP_LOGE_FOR_INVALID_FORMAT(
                context->GetNodeName(), "x", ge::GetFormatName(xFormat), "ND, NCHW, NCDHW, NHWC or NDHWC");
            return false;
    }

    inputShapes.push_back(xShape);
    inputShapes.push_back(broadcastBiasShape);
    return true;
}

ge::graphStatus BiasAddTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool BiasAddTiling::IsCapable()
{
    return true;
}

ge::graphStatus BiasAddTiling::DoOpTiling()
{
    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto biasDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);

    ge::DataType xDType = xDesc->GetDataType();
    ge::DataType biasDType = biasDesc->GetDataType();
    ge::DataType outputDType = outputDesc->GetDataType();
    OP_CHECK_IF(
        xDType != biasDType,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x and bias",
            (ge::TypeUtils::DataTypeToSerialString(xDType) + ge::TypeUtils::DataTypeToSerialString(biasDType)).c_str(),
            "The dtypes of x and bias must be the same."),
        return ge::GRAPH_FAILED);

    if (xDType != outputDType) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(xDType) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(outputDType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x and y", dtypeMsg.c_str(), "The dtypes of x and y must be the same.");
        return ge::GRAPH_FAILED;
    }

    vector<gert::Shape> inputShapes;
    OP_CHECK_IF(
        !InferBiasShape(context_, inputShapes), OP_LOGE(context_->GetNodeName(), "InferBiasShape error!"),
        return ge::GRAPH_FAILED);

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (xDType == ge::DT_BF16 || xDType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<BiasAddOp::BiasAddCompute<half>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (xDType == ge::DT_FLOAT) {
        BroadcastBaseTiling<BiasAddOp::BiasAddCompute<float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (xDType == ge::DT_INT32) {
        BroadcastBaseTiling<BiasAddOp::BiasAddCompute<int32_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (xDType == ge::DT_INT64) {
        BroadcastBaseTiling<BiasAddOp::BiasAddCompute<int64_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDType).c_str(),
            "int32, int64, float16, bf16, float");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus BiasAddTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BiasAddTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus BiasAddTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BiasAddTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BiasAddTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForBiasAdd(gert::TilingContext* context)
{
    OP_LOGD("TilingForBiasAdd", "Enter TilingForBiasAdd");
    BiasAddTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus BiasAddTilingPrepareAscendC(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<BiasAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForBiasAdd(gert::TilingParseContext* context)
{
    OP_LOGD("TilingPrepareForBiasAdd", "Enter TilingPrepareForBiasAdd");
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForBiasAdd", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = context->GetCompiledInfo<BiasAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context, "Enter BiasAddTilingPrepareAscendC");
    return BiasAddTilingPrepareAscendC(context);
}

IMPL_OP_OPTILING(BiasAdd).Tiling(TilingForBiasAdd).TilingParse<BiasAddCompileInfo>(TilingPrepareForBiasAdd);
} // namespace optiling