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
 * \file fill_tiling.cc
 * \brief
 */
#include <iostream>
#include "fill_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/fill_dag.h"

using namespace ge;
using namespace FillStruct;

namespace optiling {
constexpr uint64_t FILL_TILING_KEY_ELEMENTWISE = 101;
constexpr uint64_t FILL_WORKSPACE_RESERVE_BYTE = 16777216; // 16 * 1024 * 1024
const std::string FILLTILING_OP_NAME = "FillTiling";
constexpr uint32_t FILL_INPUT_DIMS_INDEX = 0;
constexpr uint32_t FILL_INPUT_VALUE_INDEX = 1;
constexpr uint32_t FILL_OUTPUT_Y_INDEX = 0;
constexpr int64_t MAX_DIM_NUM = 8;

ge::graphStatus FillTiling::SetTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<uint64_t>(FILL_WORKSPACE_RESERVE_BYTE);

    context_->SetTilingKey(FILL_TILING_KEY_ELEMENTWISE);
    context_->SetBlockDim(tilingData->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillTiling::CalcOutputDtype()
{
    auto inputValueDesc = context_->GetInputDesc(FILL_INPUT_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
    ge::DataType inputValueDtype = inputValueDesc->GetDataType();

    auto outputDesc = context_->GetOutputDesc(FILL_OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    this->outputDtype_ = outputDesc->GetDataType();

    OP_CHECK_IF((inputValueDtype != this->outputDtype_),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "value(input) and y(output)",
            Ops::Base::ToString(inputValueDtype) + " and " + Ops::Base::ToString(this->outputDtype_),
            "The dtypes of value(input) and y(output) must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillCheckType(ge::DataType dtype, const std::initializer_list<ge::DataType>& supportList)
{
    for (auto supportDtype : supportList) {
        if (dtype == supportDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

static bool IsDimsValid(const char_t *nodeName, const gert::Tensor *tensor, ge::DataType inputDimsDType)
{
    if (tensor == nullptr) {
        OP_LOGE(nodeName, "data tensor for dims is null");
        return false;
    }
    int64_t dimNum = tensor->GetShapeSize();
    if (dimNum < 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(nodeName, "dims(input)",
            std::to_string(dimNum),
            "The shape size of dims cannot be negative");
        return false;
    }
    if (inputDimsDType != ge::DT_INT32 && inputDimsDType != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dims(input)",
            Ops::Base::ToString(inputDimsDType),
            "Int32 or Int64");
        return false;
    }
    return true;
}

ge::graphStatus FillTiling::CheckInputDims()
{
    // Check if the dims shape is [1]
    auto dimsStorageShape = context_->GetInputShape(FILL_INPUT_DIMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimsStorageShape);
    auto dimsShape = Ops::Base::EnsureNotScalar(dimsStorageShape->GetStorageShape());
    if (dimsShape.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dims(input)",
            Ops::Base::ToString(dimsShape),
            "The shape of dims must be [1]");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *tensorDims = context_->GetInputTensor(FILL_INPUT_DIMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorDims);
    auto dimsDesc = context_->GetInputDesc(FILL_INPUT_DIMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimsDesc);
    ge::DataType inputDimsDType = dimsDesc->GetDataType();
    if (!IsDimsValid(context_->GetNodeName(), tensorDims, inputDimsDType)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static const std::initializer_list<ge::DataType> ASCEND910D_AICORE_INPUTVALUE_DTYPE_SUPPORT_LIST = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BOOL, ge::DT_INT64,
    ge::DT_INT8,  ge::DT_INT32, ge::DT_BF16};

ge::graphStatus FillTiling::CheckInputValue()
{
    // Check if the value shape size is 1
    auto valueStorageShape = context_->GetInputShape(FILL_INPUT_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueStorageShape);
    auto valueShape = Ops::Base::EnsureNotScalar(valueStorageShape->GetStorageShape());
    if (valueShape.GetShapeSize() != 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "value(input)",
            std::to_string(valueShape.GetShapeSize()),
            "The shape size of value must be 1");
        return ge::GRAPH_FAILED;
    }

    // Check if the data type
    auto valueDesc = context_->GetInputDesc(FILL_INPUT_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueDesc);
    ge::DataType inputValueDType = valueDesc->GetDataType();
    if (FillCheckType(inputValueDType, ASCEND910D_AICORE_INPUTVALUE_DTYPE_SUPPORT_LIST) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "value(input)",
            Ops::Base::ToString(inputValueDType),
            "Float, Float16, Bool, Int64, Int8, Int32 and BFloat16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(context_);
    OP_CHECK_IF(CheckInputDims() == ge::GRAPH_FAILED, OP_LOGE(context_, "check dims failed"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputValue() == ge::GRAPH_FAILED, OP_LOGE(context_, "check value failed"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(context_, "get output dtype failed"),
               return ge::GRAPH_FAILED);

    ge::graphStatus res = ge::GRAPH_FAILED;

    tilingData = context_->GetTilingData<FillTilingDataStruct>();
    if (this->outputDtype_ == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<FillDag<half>::OpDag, false>(tilingData->baseTiling);
    } else if (this->outputDtype_ == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<FillDag<float>::OpDag, false>(tilingData->baseTiling);
    } else if (this->outputDtype_ == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<FillDag<bfloat16_t>::OpDag, false>(tilingData->baseTiling);
    } else if (this->outputDtype_ == ge::DT_INT8 || this->outputDtype_ == ge::DT_BOOL) {
        res = elewiseBaseTiling.DoTiling<FillDag<int8_t>::OpDag, false>(tilingData->baseTiling);
    } else if (this->outputDtype_ == ge::DT_INT32) {
        res = elewiseBaseTiling.DoTiling<FillDag<int32_t>::OpDag, false>(tilingData->baseTiling);
    } else if (this->outputDtype_ == ge::DT_INT64) {
        res = elewiseBaseTiling.DoTiling<FillDag<int64_t>::OpDag, false>(tilingData->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y(output)",
            Ops::Base::ToString(this->outputDtype_),
            "Float16, Float, BFloat16, Int8, Bool, Int32 and Int64");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED,
        OP_LOGE(context_, "DoTiling failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus Tiling4Fill(gert::TilingContext *context)
{
    OP_LOGD(FILLTILING_OP_NAME, "Enter Tiling4Fill");
    OP_CHECK_IF(context == nullptr,
        OP_LOGE(context, "Tiling context is null"),
        return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const FillCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 走新的Asc模板tiling
    OP_LOGD(FILLTILING_OP_NAME, "Enter new FillTiling");
    FillTiling fillTiling(context);
    return fillTiling.RunTiling();
}

ge::graphStatus TilingPrepareForFill(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<FillCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(Fill).Tiling(Tiling4Fill)
    .TilingParse<FillCompileInfo>(TilingPrepareForFill)
    .InputsDataDependency({FILL_INPUT_DIMS_INDEX});
}  // namespace optiling