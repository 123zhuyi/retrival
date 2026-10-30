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
 * \file grouped_matmul_finalize_routing_quant_tiling.cpp
 * \brief
 */

#include "grouped_matmul_finalize_routing_quant_tiling.h"
#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
using namespace Ops::Transformer::OpTiling;
using namespace optiling::GroupedMatmulFinalizeRoutingArch35TilingConstant;
using namespace optiling::GmmConstant;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace optiling {

void GroupedMatmulFinalizeRoutingQuantTiling::Reset()
{
    tilingData_ = GMMFinalizeRoutingTilingData();
    OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                         context_->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
    return;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckOptionalAttr()
{
    auto *attrs = context_->GetAttrs();
    const int64_t *shareInputOffsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SHARE_INPUT_OFFSET);
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    const int64_t *outputBSPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_OUTPUT_BS);
    const int64_t *outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DTYPE);
    OP_CHECK_IF(outputBSPtr == nullptr, OP_LOGE(context_->GetNodeName(), "Attr batch is nullptr."), return false);
    int64_t shareInputOffset = shareInputOffsetPtr != nullptr ? *shareInputOffsetPtr : 0;
    int64_t groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : 1;
    int64_t outputBS = outputBSPtr != nullptr ? *outputBSPtr : 0;
    int64_t outputDtype = outputDtypePtr != nullptr ? *outputDtypePtr : 0;
    OP_CHECK_IF(
        shareInputOffset < 0 || outputBS < 0 || groupListType < 0 || outputDtype < 0,
        OP_LOGE(context_->GetNodeName(), "Attr shareInputOffset, groupListType, outputBS, outputDtype should be >=0."),
        return false);

    inputParams_.groupListType = static_cast<uint64_t>(groupListType);
    sharedInputOffset_ = static_cast<uint64_t>(shareInputOffset);
    outputBs_ = static_cast<uint64_t>(outputBS);
    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE(context_->GetNodeName(), "Attr groupListType must be 0 or 1, actual is %d.",
                        inputParams_.groupListType),
                return false);

    OP_CHECK_IF(sharedInputOffset_ > outputBs_,
                OP_LOGE(context_->GetNodeName(), "Attr sharedInputOffset (%lu) out of batch(%lu).", sharedInputOffset_,
                        outputBs_),
                return false);

    OP_CHECK_IF(outputDtype > OUT_DTYPE_BF16_INDEX,
                OP_LOGE(context_->GetNodeName(),
                        "Attr dtype only support 0(float32), 1(float16) or 2(bfloat16), actual is %d.", outputDtype),
                return false);

    OP_CHECK_IF(inputParams_.transA,
                OP_LOGE(context_->GetNodeName(), "Attr transpose_x only support false, actual is true"), return false);
    return true;
}
bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "Attrs is nullptr."), return false);
    OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_TUNING_CONFIG + 1,
                OP_LOGE(context_->GetNodeName(), "The num of attrs should be greater than %u, actual is %zu",
                        ATTR_INDEX_TUNING_CONFIG + 1, attrs->GetAttrNum()),
                return false);
    const float *shareInputWeightPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_SHARE_INPUT_WEIGHT);
    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_W);
    OP_CHECK_IF(shareInputWeightPtr == nullptr, OP_LOGE(context_->GetNodeName(), "Attr residualScale is nullptr."),
                return false);
    inputParams_.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams_.groupType = SPLIT_M;
    sharedInputWeight_ = *shareInputWeightPtr;
    OP_CHECK_IF(!CheckOptionalAttr(), OP_LOGE(context_->GetNodeName(), "Check Optional Attrs Failed."), return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input scaleDesc is nullptr."), return false);
    inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto outDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(outDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input outDesc is nullptr."), return false);
    inputParams_.cDtype = outDesc->GetDataType();
    OP_CHECK_IF(inputParams_.cDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "Output dtype should be DT_FLOAT, but now is %s ",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
                return false);

    auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    inputParams_.hasBias = !(biasStorageShape == nullptr || biasStorageShape->GetStorageShape().GetShapeSize() == 0);
    auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
    OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "Bias from tensor is not nullptr, but bias from desc is nullptr."),
                return false);
    inputParams_.biasDtype = inputParams_.hasBias ? biasDesc->GetDataType() : ge::DT_BF16;

    OP_CHECK_IF(!CheckDtype(), OP_LOGE(context_->GetNodeName(), "Required input check failed."), return false);

    OP_CHECK_IF(!CheckOptional(GROUPLIST_INDEX, "GroupList", ge::DT_INT64),
                OP_LOGE(context_->GetNodeName(), "GroupList check failed."), return false);

    OP_CHECK_IF(!CheckOptional(SHARE_INPUT_INDEX, "SharedInput", ge::DT_BF16),
                OP_LOGE(context_->GetNodeName(), "SharedInput check failed."), return false);

    OP_CHECK_IF(!CheckOptional(LOGIT_INDEX, "LogitIndex", ge::DT_FLOAT),
                OP_LOGE(context_->GetNodeName(), "LogitIndex check failed."), return false);

    if (inputParams_.aDtype != ge::DT_INT8) {
        OP_CHECK_IF(!CheckOptional(ROW_INDEX_INDEX, "RowIndex", ge::DT_INT64),
                    OP_LOGE(context_->GetNodeName(), "RowIndex check failed."), return false);
    } else if (context_->GetOptionalInputDesc(ROW_INDEX_INDEX) != nullptr) {
        auto rowIndexDtype = context_->GetOptionalInputDesc(ROW_INDEX_INDEX)->GetDataType();
        OP_CHECK_IF(!(rowIndexDtype == ge::DT_INT64 || rowIndexDtype == ge::DT_INT32),
                    OP_LOGE(context_->GetNodeName(),
                            "When inputs are DT_INT8, rowIndex dtype should be DT_INT64/DT_INT32, but now is %s ",
                            ge::TypeUtils::DataTypeToSerialString(rowIndexDtype).c_str()),
                    return false);
    }

    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckOptional(uint32_t index, const char *paramName,
                                                            ge::DataType targetDtype) const
{
    auto optionalDesc = context_->GetOptionalInputDesc(index);
    if (optionalDesc == nullptr) {
        return true;
    }
    auto realDtype = optionalDesc->GetDataType();
    OP_CHECK_IF(realDtype != targetDtype,
                OP_LOGE(context_->GetNodeName(), "%s dtype should be %s, but now is %s ", paramName,
                        ge::TypeUtils::DataTypeToSerialString(targetDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(realDtype).c_str()),
                return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::IsFp4Dtype(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT4_E2M1;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::IsFp8Dtype(ge::DataType dtype) const
{
    return (dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2);
}


bool GroupedMatmulFinalizeRoutingQuantTiling::CheckDtype() const
{
    OP_CHECK_IF(inputParams_.biasDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "The dtype of bias should be DT_BF16, but now is %s ",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
                return false);

    if (IsMicroScaling()) {
        bool a8w8 = IsFp8Dtype(inputParams_.aDtype) && IsFp8Dtype(inputParams_.bDtype);
        bool a4w4 = IsFp4Dtype(inputParams_.aDtype) && IsFp4Dtype(inputParams_.bDtype);
        OP_CHECK_IF(
            !(a8w8 || a4w4),
            OP_LOGE(context_->GetNodeName(),
                    "In mx quant mode, the expected dtype of x and weight should be \
DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2/DT_FLOAT4_E1M2/DT_FLOAT4_E2M1, but actual dtype is %s, %s.",
                    ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
            return false);
    } else {
        OP_CHECK_IF(!(inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_INT8 ||
                      inputParams_.aDtype == ge::DT_HIFLOAT8),
                    OP_LOGE(context_->GetNodeName(),
                            "In K-C/T-C quant mode, the expected dtype of x and weight should be \
DT_FLOAT8_E4M3FN/DT_INT8/DT_HIFLOAT8, but actual dtype is %s, %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    return false);

        OP_CHECK_IF(!(inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16),
                    OP_LOGE(context_->GetNodeName(),
                            "In K-C/T-C quant mode, the expected dtype of scale should be \
DT_FLOAT/DT_BF16, but actual dtype is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                    return false);

        if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) != nullptr) {
            OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                        OP_LOGE(context_->GetNodeName(),
                                "In K-C quant mode, the expected dtype of perTokenScaleDtype should be \
DT_FLOAT, but actual dtype is %s.",
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                        return false);
        }
    }

    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckDim(const gert::Shape &xShape, const gert::Shape &wShape,
                                                       const gert::StorageShape *pertokenScaleStorageShape,
                                                       const gert::Shape &scaleShape, const gert::Shape &yShape) const
{
    auto xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(xDimNum != DIM_NUM_X,
                OP_LOGE(context_->GetNodeName(), "The dimension of x must be %u, actual is %zu", DIM_NUM_X, xDimNum),
                return false);

    auto wDimNum = wShape.GetDimNum();
    OP_CHECK_IF(
        wDimNum != DIM_NUM_WEIGHT,
        OP_LOGE(context_->GetNodeName(), "The dimension of w must be %u, actual is %zu", DIM_NUM_WEIGHT, wDimNum),
        return false);

    auto yDimNum = yShape.GetDimNum();
    OP_CHECK_IF(yDimNum != DIM_NUM_Y,
                OP_LOGE(context_->GetNodeName(), "The dimension of y must be %u, actual is %zu", DIM_NUM_Y, yDimNum),
                return false);

    auto scaleDimNum = scaleShape.GetDimNum();
    if (IsMicroScaling()) {
        OP_CHECK_IF(scaleDimNum != DIM_NUM_MX_SCALE,
                    OP_LOGE(context_->GetNodeName(), "The dimension of scale must be %u, actual is %zu",
                            DIM_NUM_MX_SCALE, scaleDimNum),
                    return false);
        OP_CHECK_IF(pertokenScaleStorageShape == nullptr,
                    OP_LOGE(context_->GetNodeName(), "Input pertokenScaleStorageShape is nullptr."), return false);
        const gert::Shape &pertokenScaleShape = pertokenScaleStorageShape->GetOriginShape();
        auto pertokenScaleDimNum = pertokenScaleShape.GetDimNum();
        OP_CHECK_IF(pertokenScaleDimNum != DIM_NUM_MX_PERTOKENSCALE,
                    OP_LOGE(context_->GetNodeName(), "The dimension of pertokenScale must be %u, actual is %zu",
                            DIM_NUM_MX_PERTOKENSCALE, pertokenScaleDimNum),
                    return false);
    } else {
        OP_CHECK_IF(scaleDimNum != DIM_NUM_SCALE,
                    OP_LOGE(context_->GetNodeName(), "The dimension of scale must be %u, actual is %zu", DIM_NUM_SCALE,
                            scaleDimNum),
                    return false);
        if (pertokenScaleStorageShape != nullptr) {
            const gert::Shape &pertokenScaleShape = pertokenScaleStorageShape->GetOriginShape();
            auto pertokenScaleDimNum = pertokenScaleShape.GetDimNum();
            OP_CHECK_IF(pertokenScaleDimNum != DIM_NUM_PERTOKENSCALE,
                        OP_LOGE(context_->GetNodeName(), "The dimension of pertokenScale must be %u, actual is %zu",
                                DIM_NUM_PERTOKENSCALE, pertokenScaleDimNum),
                        return false);
        }
    }
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckFp4Shape() const
{
    bool a4w4 = IsFp4Dtype(inputParams_.aDtype) && IsFp4Dtype(inputParams_.bDtype);
    if (!a4w4) {
        return true;
    }
    OP_CHECK_IF(inputParams_.kSize % EVEN_FACTOR != 0,
                OP_LOGE(inputParams_.opName,
                        "When the dtype of x is FLOAT4, the k size should be even number, but actual k size is %lu.",
                        inputParams_.kSize),
                return false);
    // 2: mxfp4场景下不支持K轴为2
    OP_CHECK_IF(inputParams_.kSize == 2,
                OP_LOGE(inputParams_.opName, "When the dtype of x is FLOAT4, the k size should not be 2."),
                return false);
    if (!inputParams_.transB) {
        OP_CHECK_IF(
            inputParams_.nSize % EVEN_FACTOR != 0,
            OP_LOGE(inputParams_.opName,
                    "When the dtype of weight is FLOAT4 and weight is not transposed, the n size should be even number, \
but actual n size is %lu.",
                    inputParams_.nSize),
            return false);
    }
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckOptionalInputsShape()
{
    auto sharedInputDesc = context_->GetOptionalInputDesc(SHARE_INPUT_INDEX);
    sharedInputLen_ = sharedInputDesc != nullptr
                          ? context_->GetOptionalInputShape(SHARE_INPUT_INDEX)->GetStorageShape()[0]
                          : sharedInputLen_;

    OP_CHECK_IF(
        sharedInputLen_ > outputBs_,
        OP_LOGE(context_->GetNodeName(), "Input shared_input_len (%lu) out of batch(%lu).", sharedInputLen_, outputBs_),
        return false);

    OP_CHECK_IF(sharedInputOffset_ + sharedInputLen_ > outputBs_,
                OP_LOGE(context_->GetNodeName(), "SharedInputOffset + sharedInputLen (%lu) out of batch(%lu).",
                        sharedInputOffset_ + sharedInputLen_, outputBs_),
                return false);

    auto LogitDesc = context_->GetOptionalInputDesc(LOGIT_INDEX);
    OP_CHECK_IF(LogitDesc == nullptr, OP_LOGE(context_->GetNodeName(), "LogitDesc is nullptr."), return false);

    auto rowIndexDesc = context_->GetOptionalInputDesc(ROW_INDEX_INDEX);
    OP_CHECK_IF(rowIndexDesc == nullptr, OP_LOGE(context_->GetNodeName(), "RowIndexDesc is nullptr."), return false);
    rowIndex_ = context_->GetOptionalInputShape(ROW_INDEX_INDEX)->GetStorageShape()[0];

    OP_CHECK_IF(rowIndex_ > inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "Input rowIndex (%lu) out of M (%lu).", rowIndex_, inputParams_.mSize),
                return false);

    OP_CHECK_IF(outputBs_ > inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "OutputBs (%lu) out of M (%lu).", outputBs_, inputParams_.mSize),
                return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckInputsShape(const gert::Shape &xShape,
                                                               const gert::StorageShape *wStorageShape,
                                                               const gert::StorageShape *pertokenScaleStorageShape,
                                                               const gert::Shape &scaleShape,
                                                               const gert::Shape &yShape) const
{
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    OP_CHECK_IF(!CheckDim(xShape, wShape, pertokenScaleStorageShape, scaleShape, yShape),
                OP_LOGE(context_->GetNodeName(), "CheckDim failed."), return false);
    if (IsMicroScaling()) {
        OP_CHECK_IF(!CheckFp4Shape(), OP_LOGE(context_->GetNodeName(), "CheckFp4Shape failed."), return false);
    }
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "Input xStorageShape is nullptr."),
                return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
   
    auto wStorageShape = context_->GetInputShape(W_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "Input wStorageShape is nullptr."),
                return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    
    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "Input scaleStorageShape is nullptr."),
                return false);
    const gert::Shape &scaleShape = scaleStorageShape->GetOriginShape();
    
    auto pertokenScaleStorageShape = context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX);
    
    auto yStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(yStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "Output yStorageShape is nullptr."),
                return false);
    const gert::Shape &yShape = yStorageShape->GetOriginShape();

    if (!IsMicroScaling()) {
        OP_CHECK_IF(inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(inputParams_.opName, "In K-C/T-C quant mode, the format of weight must be FRACTAL_NZ"),
                    return false);
    }
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(context_->GetNodeName(), "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(context_->GetNodeName(), "SetMKN failed."), return false);
    OP_CHECK_IF(!CheckInputsShape(xShape, wStorageShape, pertokenScaleStorageShape, scaleShape, yShape),
                OP_LOGE(context_->GetNodeName(), "CheckInputsShape failed."), return false);
    OP_CHECK_IF(!CheckOptionalInputsShape(), OP_LOGE(context_->GetNodeName(), "CheckOptionalInputsShape failed."), return false);
    OP_CHECK_IF(!SetQuantModeForGMMFinalizeRouting(),
                OP_LOGE(context_->GetNodeName(), "SetQuantModeForGMMFinalizeRouting failed."), return false);
    OP_CHECK_IF(!CheckCoreNum(),
                OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);  
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckCoreNum() const
{
    auto aicNum = context_->GetCompileInfo<GroupedMatmulFinalizeRoutingCompileInfo>()->aicNum;
    auto aivNum = context_->GetCompileInfo<GroupedMatmulFinalizeRoutingCompileInfo>()->aivNum;
    OP_CHECK_IF(aicNum == 0,
               OP_LOGE(inputParams_.opName, "aicNum should be positive integer, actual is %u.", aicNum),
               return false);
    OP_CHECK_IF(aivNum != GmmConstant::CORE_RATIO * aicNum,
                OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.", aicNum, aivNum),
                return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::SetQuantModeForGMMFinalizeRouting()
{
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    }

    inputParams_.bQuantMode = optiling::QuantMode::PERCHANNEL_MODE;
    if (context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr) {
        inputParams_.aQuantMode = optiling::QuantMode::PERTOKEN_MODE;
    } else {
        inputParams_.aQuantMode = optiling::QuantMode::DEFAULT;
    }

    return true;
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoOpTiling()
{
    tilingData_.gmmFinalizeRoutingDataParams.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmFinalizeRoutingDataParams.batch = static_cast<uint32_t>(outputBs_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset = static_cast<uint32_t>(sharedInputOffset_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen = static_cast<uint32_t>(sharedInputLen_);
    tilingData_.gmmFinalizeRoutingDataParams.residualScale = static_cast<float>(sharedInputWeight_);
    tilingData_.gmmFinalizeRoutingDataParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    tilingData_.gmmFinalizeRoutingDataParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmFinalizeRoutingDataParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias ? 1 : 0);

    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulFinalizeRoutingQuantTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB));
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.matmulTiling.M = inputParams_.mSize;
    tilingData_.matmulTiling.N = inputParams_.nSize;
    tilingData_.matmulTiling.Ka = inputParams_.kSize;
    tilingData_.matmulTiling.Kb = inputParams_.kSize;
    tilingData_.matmulTiling.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.matmulTiling.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_.matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_.matmulTiling.stepM = basicTiling_.stepM;
    tilingData_.matmulTiling.stepN = basicTiling_.stepN;
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.matmulTiling.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.matmulTiling.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.matmulTiling.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }
    PrintMatmulParams();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::PostTiling()
{
    auto tilingDataSize = sizeof(GMMFinalizeRoutingTilingData);
    context_->SetBlockDim(aicoreParams_.aicNum);
    context_->SetScheduleMode(1);
    OP_CHECK_IF(tilingDataSize % sizeof(uint64_t) != 0,
                OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8", tilingDataSize),
                return ge::GRAPH_FAILED);
    error_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(inputParams_.opName, "Memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulFinalizeRoutingQuantTiling::PrintMatmulParams()
{
    int32_t enable = CheckLogLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }
    std::ostringstream oss;
    oss << "GMM matmul tiling: M = " << tilingData_.matmulTiling.M << ", N = " << tilingData_.matmulTiling.N
        << ", Ka = " << tilingData_.matmulTiling.Ka << ", Kb = " << tilingData_.matmulTiling.Kb
        << ", usedCoreNum = " << tilingData_.matmulTiling.usedCoreNum << ", baseM = " << tilingData_.matmulTiling.baseM
        << ", baseN = " << tilingData_.matmulTiling.baseN << ", baseK = " << tilingData_.matmulTiling.baseK;
    OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
}

void GroupedMatmulFinalizeRoutingQuantTiling::PrintQuantParams()
{
    int32_t enable = CheckLogLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }

    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << tilingData_.gmmFinalizeRoutingDataParams.groupNum
        << ", groupListType = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.groupListType)
        << ", batch = " << tilingData_.gmmFinalizeRoutingDataParams.batch
        << ", sharedInputOffset = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset
        << ", sharedInputLen = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen
        << ", residualScale = " << tilingData_.gmmFinalizeRoutingDataParams.residualScale
        << ", aQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.aQuantMode
        << ", bQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.bQuantMode
        << ", hasBias = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.hasBias);
    OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulFinalizeRouting, GroupedMatmulFinalizeRoutingQuantTiling, 1);
} // namespace optiling