/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file confusion_transpose_d_tiling_arch35.cpp
 * \brief
 */

#include "confusion_transpose_d_tiling_arch35.h"
#include "../../../transpose/op_host/arch35/transpose_tiling_arch35.h"

namespace optiling {

static const std::vector<ge::DataType> SUPPORT_DTYPE_ND = {ge::DT_INT8,    ge::DT_INT16,  ge::DT_INT32,  ge::DT_INT64,
                                                           ge::DT_UINT8,   ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64,
                                                           ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_BF16};

ge::graphStatus ConfusionTransposeDTiling::GetParameters()
{
    auto xStorageShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xStorageShape);
    auto xShape = xStorageShape->GetStorageShape();
    paramInfo_.xShape = xShape;

    auto xDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xDesc);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xDtype = xDesc->GetDataType();
    paramInfo_.xDtype = xDtype;
    paramInfo_.xFormat = xFormat;

    auto yStorageShape = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yStorageShape);
    auto yShape = yStorageShape->GetStorageShape();
    paramInfo_.yShape = yShape;

    auto yDesc = tilingContext_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yDesc);
    auto yDtype = yDesc->GetDataType();
    auto yFormat = yDesc->GetFormat().GetStorageFormat();
    paramInfo_.yDtype = yDtype;
    paramInfo_.yFormat = yFormat;

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    auto permPtr = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, permPtr);
    paramInfo_.perm.SetDimNum(permPtr->GetSize());
    for (int64_t i = 0; i < int64_t(paramInfo_.perm.GetDimNum()); i++) {
        paramInfo_.perm[i] = permPtr->GetData()[i];
    }

    auto shapePtr = attrs->GetListInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, shapePtr);
    paramInfo_.shape.SetDimNum(shapePtr->GetSize());
    for (int64_t i = 0; i < int64_t(paramInfo_.shape.GetDimNum()); i++) {
        paramInfo_.shape[i] = shapePtr->GetData()[i];
    }

    auto transposeFirstPtr = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, transposeFirstPtr);
    paramInfo_.transpose_first = *transposeFirstPtr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingFormatAndDatatype()
{
    if (paramInfo_.xFormat != ge::FORMAT_FRACTAL_NZ) {
        paramInfo_.xFormat = ge::FORMAT_ND;
    }
    if (paramInfo_.xFormat == ge::FORMAT_ND) {
        OP_CHECK_IF(
            std::find(SUPPORT_DTYPE_ND.begin(), SUPPORT_DTYPE_ND.end(), paramInfo_.xDtype) == SUPPORT_DTYPE_ND.end(),
            OP_LOGE_FOR_INVALID_DTYPE(
                tilingContext_->GetNodeName(), "x", Ops::Base::ToString(paramInfo_.xDtype).c_str(),
                "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT16, FLOAT and BF16"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(tilingContext_->GetNodeName(), "x", Ops::Base::ToString(paramInfo_.xFormat).c_str(),
                                   "ND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingInputAndOutput()
{
    if (paramInfo_.xDtype != paramInfo_.yDtype) {
        std::string dtypeMsg = Ops::Base::ToString(paramInfo_.xDtype) + " and " +
                               Ops::Base::ToString(paramInfo_.yDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext_->GetNodeName(), "x and y", dtypeMsg.c_str(),
                                               "The dtypes of x and y must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingDimNd()
{
    if (paramInfo_.xShape.GetDimNum() > MAX_DIM_ND || paramInfo_.xShape.GetDimNum() < MIN_DIM_ND) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_ND) + " and " + std::to_string(MAX_DIM_ND);
        OP_LOGE_FOR_INVALID_SHAPEDIM(tilingContext_->GetNodeName(), "x",
                                     std::to_string(paramInfo_.xShape.GetDimNum()).c_str(), dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.yShape.GetDimNum() > MAX_DIM_ND || paramInfo_.yShape.GetDimNum() < MIN_DIM_ND) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_ND) + " and " + std::to_string(MAX_DIM_ND);
        OP_LOGE_FOR_INVALID_SHAPEDIM(tilingContext_->GetNodeName(), "y",
                                     std::to_string(paramInfo_.yShape.GetDimNum()).c_str(), dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.shape.GetDimNum() > MAX_DIM_ND || paramInfo_.shape.GetDimNum() < MIN_DIM_ND) {
        std::string dimExpectRange = "between " + std::to_string(MIN_DIM_ND) + " and " + std::to_string(MAX_DIM_ND);
        OP_LOGE_FOR_INVALID_SHAPEDIM(tilingContext_->GetNodeName(), "shape",
                                     std::to_string(paramInfo_.shape.GetDimNum()).c_str(), dimExpectRange.c_str());
        return ge::GRAPH_FAILED;
    }

    if (paramInfo_.transpose_first) {
        if (paramInfo_.perm.GetDimNum() != paramInfo_.xShape.GetDimNum()) {
            std::string dimMsg = std::to_string(paramInfo_.xShape.GetDimNum()) + " and " +
                                 std::to_string(paramInfo_.perm.GetDimNum());
            std::string reasonMsg = "The shape dimensions of perm and x must be the same, when input x need to be "
                                    "transposed first";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(tilingContext_->GetNodeName(), "x and perm", dimMsg.c_str(),
                                                      reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (paramInfo_.perm.GetDimNum() != paramInfo_.shape.GetDimNum()) {
            std::string dimMsg = std::to_string(paramInfo_.perm.GetDimNum()) + " and " +
                                 std::to_string(paramInfo_.shape.GetDimNum());
            std::string reasonMsg = "The shape dimensions of parameter perm and parameter shape must be the same, when "
                                    "input x does not need "
                                    "to be transposed first";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(tilingContext_->GetNodeName(), "perm and shape", dimMsg.c_str(),
                                                      reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingPerm()
{
    int64_t checkPerm = 0;
    for (int64_t i = 0; i < int64_t(paramInfo_.perm.GetDimNum()); i++) {
        if (paramInfo_.perm[i] < int64_t(paramInfo_.perm.GetDimNum())) {
            checkPerm += (1l << paramInfo_.perm[i]);
        }
    }
    if (checkPerm != (1l << paramInfo_.perm.GetDimNum()) - 1) {
        OP_LOGE_FOR_INVALID_VALUE(tilingContext_->GetNodeName(), "perm", Ops::Base::ToString(paramInfo_.perm).c_str(),
                                  "a permutation of 0,1, ..., dim - 1");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifyingProdAndPositive()
{
    int64_t prodX = 1;
    int64_t prodY = 1;
    int64_t prodShape = 1;
    bool xPositive = true;
    bool yPositive = true;
    bool shapePositive = true;
    for (int64_t i = 0; i < int64_t(paramInfo_.xShape.GetDimNum()); i++) {
        xPositive &= (paramInfo_.xShape.GetDim(i) > 0);
        prodX *= paramInfo_.xShape.GetDim(i);
    }
    for (int64_t i = 0; i < int64_t(paramInfo_.yShape.GetDimNum()); i++) {
        yPositive &= (paramInfo_.yShape.GetDim(i) > 0);
        prodY *= paramInfo_.yShape.GetDim(i);
    }
    for (int64_t i = 0; i < int64_t(paramInfo_.shape.GetDimNum()); i++) {
        shapePositive &= (paramInfo_.shape.GetDim(i) > 0);
        prodShape *= paramInfo_.shape.GetDim(i);
    }

    OP_CHECK_IF(xPositive == false,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "x",
                                                      Ops::Base::ToString(paramInfo_.xShape).c_str(),
                                                      "All axes of x must be positive numbers"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yPositive == false,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "y",
                                                      Ops::Base::ToString(paramInfo_.yShape).c_str(),
                                                      "All axes of output y must be positive numbers"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shapePositive == false,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "shape",
                                                      Ops::Base::ToString(paramInfo_.shape).c_str(),
                                                      "All axes of parameter shape must be positive numbers"),
                return ge::GRAPH_FAILED);

    if (prodX > prodY || prodX != prodShape) {
        std::string shapeMsg = std::to_string(prodX) + ", " + std::to_string(prodY) + " and " +
                               std::to_string(prodShape);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            tilingContext_->GetNodeName(), "x, y and shape", shapeMsg.c_str(),
            "The shape sizes of parameter x and parameter shape must be the same, and the shape size of x must less "
            "than or equal to that of y");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionTransposeDTiling::ParametersVerifying()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start ConfusionTransposeDTiling ParametersVerifying.");

    if (GetParameters() == ge::GRAPH_FAILED || ParametersVerifyingFormatAndDatatype() == ge::GRAPH_FAILED ||
        ParametersVerifyingInputAndOutput() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (ParametersVerifyingDimNd() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (ParametersVerifyingPerm() == ge::GRAPH_FAILED || ParametersVerifyingProdAndPositive() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(tilingContext_->GetNodeName(),
            "x shape is %s, output shape is %s, shape is %s, perm is %s, transpose_first is %d",
            Ops::Base::ToString(paramInfo_.xShape).c_str(), Ops::Base::ToString(paramInfo_.yShape).c_str(),
            Ops::Base::ToString(paramInfo_.shape).c_str(), Ops::Base::ToString(paramInfo_.perm).c_str(),
            paramInfo_.transpose_first);
    return ge::GRAPH_SUCCESS;
}

void ConfusionTransposeDTiling::ProcessShapeInfoForNd(ShapeInfo& shapeInfo)
{
    if (paramInfo_.transpose_first) {
        shapeInfo.inShapeSize = paramInfo_.xShape.GetDimNum();
        shapeInfo.outShapeSize = paramInfo_.xShape.GetDimNum();
        for (int64_t i = 0; i < shapeInfo.inShapeSize; i++) {
            shapeInfo.inShape[i] = paramInfo_.xShape[i];
            shapeInfo.outShape[i] = paramInfo_.xShape[paramInfo_.perm[i]];
        }
        shapeInfo.dim = paramInfo_.xShape.GetDimNum();
        shapeInfo.origDim = paramInfo_.xShape.GetDimNum();
    } else {
        shapeInfo.inShapeSize = paramInfo_.shape.GetDimNum();
        shapeInfo.outShapeSize = paramInfo_.shape.GetDimNum();
        for (int64_t i = 0; i < shapeInfo.inShapeSize; i++) {
            shapeInfo.inShape[i] = paramInfo_.shape[i];
            shapeInfo.outShape[i] = paramInfo_.shape[paramInfo_.perm[i]];
        }
        shapeInfo.dim = paramInfo_.shape.GetDimNum();
        shapeInfo.origDim = paramInfo_.shape.GetDimNum();
    }
    shapeInfo.permSize = paramInfo_.perm.GetDimNum();
    for (int64_t i = 0; i < shapeInfo.permSize; i++) {
        shapeInfo.perm[i] = paramInfo_.perm[i];
    }
}

void ConfusionTransposeDTiling::ProcessShapeInfo(ShapeInfo& shapeInfo)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start ConfusionTransposeDTiling ProcessShapeInfo.");
    shapeInfo.eleLenInBytes = ge::GetSizeByDataType(paramInfo_.xDtype);
    OP_LOGD(tilingContext_->GetNodeName(), "Is Nd tiling.");
    ProcessShapeInfoForNd(shapeInfo);
}

ge::graphStatus ConfusionTransposeDTilingForAscendC(gert::TilingContext* context, uint64_t coreNum, uint64_t ubSize)
{
    OP_LOGD(context->GetNodeName(), "Start ConfusionTransposeDTilingForAscendC.");
    ConfusionTransposeDTiling tilingObject(context);

    OP_CHECK_IF(tilingObject.ParametersVerifying() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ConfusionTransposeDTiling failed to verify params!"),
                return ge::GRAPH_FAILED);

    // construct an equivalent Transpose inputShapeInfo
    ShapeInfo inputShapeInfo;
    try {
        tilingObject.ProcessShapeInfo(inputShapeInfo);
    } catch (const std::invalid_argument& e) {
        OP_LOGE(context->GetNodeName(), "Failed to set shape info, reason: %s", e.what());
        return ge::GRAPH_FAILED;
    }

    ConfusionTransposeDTilingData tilingData;
    ConfusionTransposeDCompileInfo compileInfo;
    compileInfo.transposeCompilerInfo.coreNum = coreNum;
    compileInfo.transposeCompilerInfo.ubSize = ubSize;

    TransposeNddmaTiling transposeTilingObject(context);
    OP_CHECK_IF((transposeTilingObject.TilingForReleatedTranspose(context, &tilingData.transposeOpTiling,
                                                                  &compileInfo.transposeCompilerInfo,
                                                                  inputShapeInfo) == ge::GRAPH_FAILED),
                OP_LOGE(context->GetNodeName(), "Transpose Tiling failed"), return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context->GetNodeName(), "ConfusionTransposeDTilingForAscendC success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForConfusionTransposeD(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingForConfusionTransposeD");

    OP_LOGD(context->GetNodeName(), "in ascendc");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "ConfusionTransposeD Op GetHardwareInfo Failed, coreNum: %lu.", coreNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF((ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "ConfusionTransposeD Op GetHardwareInfo Failed, ubSize: %lu.", ubSize),
                return ge::GRAPH_FAILED);

    return ConfusionTransposeDTilingForAscendC(context, coreNum, ubSize);
}

static ge::graphStatus TilingPrepareConfusionTransposeDForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepareConfusionTransposeDForAscendC");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (ci->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "ConfusionTransposeD Op GetHardwareInfo Failed, coreNum:%ld.", ci->coreNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (ci->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "ConfusionTransposeD Op GetHardwareInfo Failed, ubSize:%ld.", ci->ubSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "ConfusionTransposeD Op get coreNum:%lu, ubSize:%lu.", ci->coreNum, ci->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForConfusionTransposeD(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepareForConfusionTransposeD.");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    TilingPrepareConfusionTransposeDForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ConfusionTransposeD)
    .Tiling(TilingForConfusionTransposeD)
    .TilingParse<ConfusionTransposeDCompileInfo>(TilingPrepareForConfusionTransposeD);

} // namespace optiling
