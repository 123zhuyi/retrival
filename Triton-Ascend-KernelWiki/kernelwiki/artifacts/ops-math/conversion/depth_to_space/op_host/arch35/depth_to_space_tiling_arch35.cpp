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
 * \file depth_to_space_tiling_arch35.cpp
 * \brief tiling for DepthToSpace
 */
#include "depth_to_space_tiling_arch35.h"

namespace optiling {

namespace DepthToSpace {
ge::graphStatus DepthToSpaceTiling::ParametersVerifying()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DepthToSpaceTiling ParametersVerifying.");

    auto xStorageShape = tilingContext_->GetInputShape(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xStorageShape);
    auto xShape = xStorageShape->GetStorageShape();
    auto xDimNum = xShape.GetDimNum();
    paramInfo_.xShape = xShape;

    auto yStorageShape = tilingContext_->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yStorageShape);
    auto yShape = yStorageShape->GetStorageShape();
    auto yDimNum = yShape.GetDimNum();

    auto xDesc = tilingContext_->GetInputDesc(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xDesc);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xDtype = xDesc->GetDataType();
    paramInfo_.xDtype = xDtype;

    auto yDesc = tilingContext_->GetOutputDesc(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yDesc);
    auto yFormat = yDesc->GetFormat().GetStorageFormat();
    if (CheckFormatAndShape(xDimNum, yDimNum, xFormat, yFormat) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    auto blockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, blockSizePtr);
    paramInfo_.blockSize = *blockSizePtr;

    const char* modePtr = attrs->GetAttrPointer<char>(ATTR_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, modePtr);
    paramInfo_.modePtr = modePtr;

    const char* dataFormatPtr = attrs->GetAttrPointer<char>(ATTR_DEPTH_DATA_FORMAT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, dataFormatPtr);
    paramInfo_.dataFormatPtr = dataFormatPtr;

    if (CheckAttrValues(xFormat) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DepthToSpaceTiling::CheckFormatAndShape(int64_t xDimNum, int64_t yDimNum, ge::Format xFormat,
                                                        ge::Format yFormat)
{
    if (xDimNum != DIM_NUM || yDimNum != DIM_NUM) {
        std::string incorrectDims = std::to_string(xDimNum) + " and " + std::to_string(yDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(tilingContext_->GetNodeName(), "x and y", incorrectDims.c_str(),
                                                  "The shape of x and y must be 4D");
        return ge::GRAPH_FAILED;
    }

    if (xFormat != yFormat || (xFormat != ge::FORMAT_NCHW && xFormat != ge::FORMAT_NHWC)) {
        std::string incorrectFormats = Ops::Base::ToString(xFormat) + " and " + Ops::Base::ToString(yFormat);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            tilingContext_->GetNodeName(), "x and y", incorrectFormats.c_str(),
            "The formats of x and y support only the following combinations: NCHW and NHWC, and the formats of x and y "
            "must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DepthToSpaceTiling::CheckAttrValues(ge::Format xFormat)
{
    ge::Format geFormat = ge::FORMAT_NCHW;
    if (strcmp(paramInfo_.dataFormatPtr, "NCHW") == 0) {
        geFormat = ge::FORMAT_NCHW;
    } else if (strcmp(paramInfo_.dataFormatPtr, "NHWC") == 0) {
        geFormat = ge::FORMAT_NHWC;
    } else {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(tilingContext_->GetNodeName(), "data_format", paramInfo_.dataFormatPtr,
                                                "The formats of data_format must be NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }

    if (xFormat != geFormat) {
        std::string incorrectValues = std::string(paramInfo_.dataFormatPtr) + " and " + Ops::Base::ToString(xFormat);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(tilingContext_->GetNodeName(), "data_format and x",
                                               incorrectValues.c_str(),
                                               "The value of data_format must be equal to that of x");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((paramInfo_.blockSize < 2),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    tilingContext_->GetNodeName(), "block_size", std::to_string(paramInfo_.blockSize).c_str(),
                    "The value of block_size must be a positive integer greater than or equal to 2"),
                return ge::GRAPH_FAILED);

    auto depth = paramInfo_.xShape.GetDim(xFormat == ge::FORMAT_NCHW ? 1 : 3);
    OP_CHECK_IF(
        (depth % (paramInfo_.blockSize * paramInfo_.blockSize) != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            tilingContext_->GetNodeName(), "block_size", std::to_string(paramInfo_.blockSize).c_str(),
            "The value of block_size must be a divisor such that depth is divisible by the square of block_size"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((strcmp(paramInfo_.modePtr, "DCR") != 0 && strcmp(paramInfo_.modePtr, "CRD") != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "mode", paramInfo_.modePtr,
                                                      "The value of mode can only be DCR or CRD"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void DepthToSpaceTiling::SetPermAndOutShape(ShapeInfo& shapeInfo, const int64_t* perm)
{
    for (int64_t i = 0; i < DIM_SIX; i++) {
        shapeInfo.perm[i] = perm[i];
        shapeInfo.outShape[i] = shapeInfo.inShape[shapeInfo.perm[i]];
    }
}

void DepthToSpaceTiling::ProcessShapeInfo(ShapeInfo& shapeInfo)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DepthToSpaceTiling ProcessShapeInfo.");
    shapeInfo.permSize = DIM_SIX;
    shapeInfo.eleLenInBytes = ge::GetSizeByDataType(paramInfo_.xDtype);
    shapeInfo.inShapeSize = DIM_SIX;
    shapeInfo.outShapeSize = DIM_SIX;
    shapeInfo.dim = DIM_SIX;
    shapeInfo.origDim = DIM_SIX;
    if (strcmp(paramInfo_.dataFormatPtr, "NHWC") == 0 && strcmp(paramInfo_.modePtr, "DCR") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.xShape[DIM_ONE];
        shapeInfo.inShape[DIM_TWO] = paramInfo_.xShape[DIM_TWO];
        shapeInfo.inShape[DIM_THREE] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.xShape[DIM_THREE] / (paramInfo_.blockSize * paramInfo_.blockSize);
        SetPermAndOutShape(shapeInfo, nhwcDcrPerm_);
    } else if (strcmp(paramInfo_.dataFormatPtr, "NCHW") == 0 && strcmp(paramInfo_.modePtr, "DCR") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_TWO] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_THREE] = paramInfo_.xShape[DIM_ONE] / (paramInfo_.blockSize * paramInfo_.blockSize);
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.xShape[DIM_TWO];
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.xShape[DIM_THREE];
        SetPermAndOutShape(shapeInfo, nchwDcrPerm_);
    } else if (strcmp(paramInfo_.dataFormatPtr, "NHWC") == 0 && strcmp(paramInfo_.modePtr, "CRD") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.xShape[DIM_ONE];
        shapeInfo.inShape[DIM_TWO] = paramInfo_.xShape[DIM_TWO];
        shapeInfo.inShape[DIM_THREE] = paramInfo_.xShape[DIM_THREE] / (paramInfo_.blockSize * paramInfo_.blockSize);
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.blockSize;
        SetPermAndOutShape(shapeInfo, crdPerm_);
    } else if (strcmp(paramInfo_.dataFormatPtr, "NCHW") == 0 && strcmp(paramInfo_.modePtr, "CRD") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.xShape[DIM_ONE] / (paramInfo_.blockSize * paramInfo_.blockSize);
        shapeInfo.inShape[DIM_TWO] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_THREE] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.xShape[DIM_TWO];
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.xShape[DIM_THREE];
        SetPermAndOutShape(shapeInfo, crdPerm_);
    } else {
        std::string incorrectFormats = std::string(paramInfo_.dataFormatPtr) + "+" + paramInfo_.modePtr;
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            tilingContext_->GetNodeName(), "data_format and mode", incorrectFormats.c_str(),
            "The formats of these parameters support only the following combinations: (NCHW+DCR), (NCHW+CRD), "
            "(NHWC+DCR), (NHWC+CRD)");
    }
}

ge::graphStatus DepthToSpaceTilingForAscendC(gert::TilingContext* context,
                                             const TransposeCompilerInfo* transposeCompileInfo)
{
    OP_LOGD(context->GetNodeName(), "Start DepthToSpaceTilingForAscendC.");
    DepthToSpaceTiling tilingObject(context);

    OP_CHECK_IF(tilingObject.ParametersVerifying() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "DepthToSpaceTiling failed to verify params!"),
                return ge::GRAPH_FAILED);

    // construct an equivalent Transpose inputShapeInfo
    ShapeInfo inputShapeInfo;
    tilingObject.ProcessShapeInfo(inputShapeInfo);

    DepthToSpaceTilingData tilingData;
    DepthToSpaceCompileInfo compileInfo;
    compileInfo.transposeCompilerInfo.coreNum = transposeCompileInfo->coreNum;
    compileInfo.transposeCompilerInfo.ubSize = transposeCompileInfo->ubSize;

    TransposeNddmaTiling transposeTilingObject(context);
    OP_CHECK_IF((transposeTilingObject.TilingForReleatedTranspose(context, &tilingData.transposeOpTiling,
                                                                  &compileInfo.transposeCompilerInfo,
                                                                  inputShapeInfo) == ge::GRAPH_FAILED),
                OP_LOGE(context->GetNodeName(), "Transpose Tiling failed"), return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context->GetNodeName(), "DepthToSpaceTilingForAscendC success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForDepthToSpace(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingForDepthToSpace");
    auto compile_info = context->GetCompileInfo<TransposeCompilerInfo>();
    return DepthToSpaceTilingForAscendC(context, compile_info);
}

static ge::graphStatus TilingPrepareForRelatedToTranspose(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepareForRelatedToTranspose.");
    TilingPrepareTransposeForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DepthToSpace)
    .Tiling(TilingForDepthToSpace)
    .TilingParse<TransposeCompilerInfo>(TilingPrepareForRelatedToTranspose);

} // namespace DepthToSpace
} // namespace optiling
