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
 * \file    select_tiling.cpp
 * \brief   select_tiling source file
 */

#include "select_tiling.h"
#include <graph/utils/type_utils.h>
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/select/op_kernel/arch35/select_dag.h"
#include "math/select/op_kernel/arch35/select_struct.h"
#include "log/log.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr uint64_t SELECT_COMMON_TILING_PRIORITY = 1;
static constexpr int32_t CONDITION_IDX = 0;
static constexpr int32_t X1_IDX = 1;
static constexpr int32_t X2_IDX = 2;
static constexpr int32_t Y_IDX = 0;

template <typename T>
std::string Shape2String(const T& shape)
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

ge::graphStatus SelectTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool SelectTiling::IsCapable()
{
    return true;
}

ge::graphStatus SelectCheckInputDtype(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(CONDITION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType input0DType = inputDesc->GetDataType();

    auto input1Desc = context->GetInputDesc(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();

    auto input2Desc = context->GetInputDesc(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input2Desc);
    ge::DataType input2DType = input2Desc->GetDataType();

    auto outputYDesc = context->GetOutputDesc(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYDesc);
    ge::DataType outputDtype = outputYDesc->GetDataType();
    if ((input0DType != ge::DT_BOOL) || (input1DType != input2DType) || (outputDtype != input1DType)) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(input0DType) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(input1DType) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(input2DType) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "condition, x1, x2 and y",
            dtypesStr.c_str(), "The dtype of Condition must be bool, and the dtypes of x1, x2 and y must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferSelectShape(gert::TilingContext* context, vector<gert::Shape>& inputShapes)
{
    auto conditionStorageShape = context->GetInputShape(CONDITION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, conditionStorageShape);
    auto& conditionShape = Ops::Base::EnsureNotScalar(conditionStorageShape->GetStorageShape());
    int64_t conditionDimNum = conditionShape.GetDimNum();

    auto x1StorageShape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1StorageShape);
    auto& x1Shape = Ops::Base::EnsureNotScalar(x1StorageShape->GetStorageShape());
    int64_t x1DimNum = x1Shape.GetDimNum();

    auto x2StorageShape = context->GetInputShape(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2StorageShape);
    auto& x2Shape = Ops::Base::EnsureNotScalar(x2StorageShape->GetStorageShape());

    if (x1Shape != x2Shape) {
        std::string x1ShapeStr = Ops::Base::ToString(x1Shape);
        std::string x2ShapeStr = Ops::Base::ToString(x2Shape);
        std::string shapesStr = x1ShapeStr + " and " + x2ShapeStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1 and x2",
            shapesStr.c_str(), "The shapes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }
        
    int64_t conditionResize = x1DimNum - conditionDimNum;
    OP_CHECK_IF(conditionResize < 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "condition and x1",
            (std::to_string(conditionDimNum) + " and " + std::to_string(x1DimNum)).c_str(),
            "The shape dim of condition should not be greater than the shape dim of x1"),
        return ge::GRAPH_FAILED);

    gert::Shape conditionNewShape;
    conditionNewShape.SetDimNum(0);

    for (int64_t i = 0; i < conditionDimNum; i++) {
        conditionNewShape.AppendDim(conditionShape.GetDim(i));
    }
    if (conditionResize > 0) {
        for (int64_t i = 0; i < conditionResize; i++) {
            conditionNewShape.AppendDim(1);
        }
    }
    inputShapes.push_back(conditionNewShape);
    inputShapes.push_back(x1Shape);
    inputShapes.push_back(x2Shape);  
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SelectTiling::DoOpTiling()
{
    OP_CHECK_IF(SelectCheckInputDtype(context_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "SelectCheckInputDtype error!"),
        return ge::GRAPH_FAILED);

    vector<gert::Shape> inputShapes;
    OP_CHECK_IF(InferSelectShape(context_, inputShapes) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "InferSelectShape error!"),
        return ge::GRAPH_FAILED);

    auto input1Desc = context_->GetInputDesc(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input1DType == ge::DT_INT64) {
        Ops::Base::BroadcastBaseTiling<SelectOp::Select<int64_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), 0);
    } else if (input1DType == ge::DT_INT8) {
        Ops::Base::BroadcastBaseTiling<SelectOp::Select<int8_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), 0);
    } else if (input1DType == ge::DT_UINT8 || input1DType == ge::DT_BOOL) {
        Ops::Base::BroadcastBaseTiling<SelectOp::Select<uint8_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), 0);
    } else if (input1DType == ge::DT_INT32) {
        Ops::Base::BroadcastBaseTiling<SelectOp::Select<int32_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), 0);
    } else if (input1DType == ge::DT_FLOAT16 || input1DType == ge::DT_BF16) {
        Ops::Base::BroadcastBaseTiling<SelectOp::Select<Ops::Base::half>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), 0);
    } else if (input1DType == ge::DT_FLOAT) {
        Ops::Base::BroadcastBaseTiling<SelectOp::Select<float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), 0);
    } else {
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(input1DType);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
            "int64, int32, int8, uint8, float16, bf16, fp32 or bool");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus SelectTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SelectTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus SelectTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SelectTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SelectTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSelect(gert::TilingContext* context)
{
    OP_LOGD("SelectTiling", "Enter TilingForSelect");
    if (context == nullptr) {
        OP_LOGE("SelectTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const SelectCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc SelectTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus SelectTilingPrepareAscendC(gert::TilingParseContext* context)
{
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<SelectCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

 ge::graphStatus TilingPrepareForSelect(gert::TilingParseContext* context)
{
    OP_LOGD("TilingPrepareForSelect", "Enter TilingPrepareForSelect");
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForSelect", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = context->GetCompiledInfo<SelectCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter SelectTilingPrepareAscendC");
    return SelectTilingPrepareAscendC(context);
}
 
 IMPL_OP_OPTILING(Select)
     .Tiling(TilingForSelect)
     .TilingParse<SelectCompileInfo>(TilingPrepareForSelect);
 REGISTER_OPS_TILING_TEMPLATE(Select, SelectTiling, SELECT_COMMON_TILING_PRIORITY);
 }   // namespace optiling 