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
 * \file space_to_depth_tiling_arch35.cpp
 * \brief
 */

#include "util/platform_util.h"
#include "space_to_depth_tiling_arch35.h"
#include "conversion/transpose/op_host/arch35/transpose_tiling_base.h"

namespace optiling {
ge::graphStatus SpaceToDepthTiling::CheckShapeAndFormat(ge::Format& xFormat)
{
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
    xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xDtype = xDesc->GetDataType();
    paramInfo_.xDtype = xDtype;

    auto yDesc = tilingContext_->GetOutputDesc(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, yDesc);
    auto yFormat = yDesc->GetFormat().GetStorageFormat();

    // limit input output dim is 4D
    if (xDimNum != DIM_NUM || yDimNum != DIM_NUM) {
        std::string dimsStr = std::to_string(xDimNum) + " and " + std::to_string(yDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(tilingContext_->GetNodeName(), "x and y", dimsStr.c_str(),
                                                  "The shape of x and y must be 4D");
        return ge::GRAPH_FAILED;
    }

    // limit input output format is same and limit NCHW and NHWC
    if (xFormat != yFormat || (xFormat != ge::FORMAT_NCHW && xFormat != ge::FORMAT_NHWC)) {
        std::string formatMsg = std::to_string(xFormat) + " and " + std::to_string(yFormat);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            tilingContext_->GetNodeName(), "x and y", formatMsg.c_str(),
            "The formats of x and y must be the same and only support NCHW and NHWC");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToDepthTiling::CheckAttrsAndBlockSize(ge::Format xFormat)
{
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    auto blockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, blockSizePtr);
    paramInfo_.blockSize = *blockSizePtr;

    const char* dataFormatPtr = attrs->GetAttrPointer<char>(ATTR_SPACE_DATA_FORMAT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, dataFormatPtr);
    paramInfo_.dataFormatPtr = dataFormatPtr;

    // limit input format is same as attr format
    ge::Format geFormat = ge::FORMAT_NCHW;
    if (strcmp(dataFormatPtr, "NCHW") == 0) {
        geFormat = ge::FORMAT_NCHW;
    } else if (strcmp(dataFormatPtr, "NHWC") == 0) {
        geFormat = ge::FORMAT_NHWC;
    } else {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(tilingContext_->GetNodeName(), "data_format", dataFormatPtr,
                                                "The value of data_format can only be NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }

    if (xFormat != geFormat) {
        std::string formatMsg = std::to_string(xFormat) + " and " + std::to_string(static_cast<int64_t>(geFormat));
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(tilingContext_->GetNodeName(), "data_format and x", formatMsg.c_str(),
                                                "The formats of data_format and x must be the same");
        return ge::GRAPH_FAILED;
    }

    // limit blockSize positive integer
    OP_CHECK_IF((*blockSizePtr <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "block_size",
                                                      std::to_string(*blockSizePtr).c_str(),
                                                      "The value of block_size must be a positive integer"),
                return ge::GRAPH_FAILED);

    // H % block_size == 0
    auto h = paramInfo_.xShape.GetDim(xFormat == ge::FORMAT_NCHW ? 2 : 1);
    OP_CHECK_IF((h % *blockSizePtr != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "block_size",
                                                      std::to_string(*blockSizePtr).c_str(),
                                                      "H must be divisible by block_size"),
                return ge::GRAPH_FAILED);

    // W % block_size == 0
    auto w = paramInfo_.xShape.GetDim(xFormat == ge::FORMAT_NCHW ? 3 : 2);
    OP_CHECK_IF((w % *blockSizePtr != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "block_size",
                                                      std::to_string(*blockSizePtr).c_str(),
                                                      "W must be divisible by block_size"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToDepthTiling::ParametersVerifying()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start SpaceToDepthTiling ParametersVerifying.");

    ge::Format xFormat = ge::FORMAT_NCHW;
    if (CheckShapeAndFormat(xFormat) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckAttrsAndBlockSize(xFormat);
}

void SpaceToDepthTiling::ProcessShapeInfo(ShapeInfo& shapeInfo)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start SpaceToDepthTiling ProcessShapeInfo.");
    shapeInfo.permSize = DIM_SIX;
    shapeInfo.eleLenInBytes = ge::GetSizeByDataType(paramInfo_.xDtype);
    shapeInfo.inShapeSize = DIM_SIX;
    shapeInfo.outShapeSize = DIM_SIX;
    shapeInfo.dim = DIM_SIX;
    shapeInfo.origDim = DIM_SIX;
    if (strcmp(paramInfo_.dataFormatPtr, "NHWC") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.xShape[DIM_ONE] / paramInfo_.blockSize;
        shapeInfo.inShape[DIM_TWO] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_THREE] = paramInfo_.xShape[DIM_TWO] / paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.xShape[DIM_THREE];
        for (int64_t i = 0; i < DIM_SIX; i++) {
            shapeInfo.perm[i] = nhwcPerm_[i];
            shapeInfo.outShape[i] = shapeInfo.inShape[shapeInfo.perm[i]];
        }
    } else if (strcmp(paramInfo_.dataFormatPtr, "NCHW") == 0) {
        shapeInfo.inShape[DIM_ZERO] = paramInfo_.xShape[DIM_ZERO];
        shapeInfo.inShape[DIM_ONE] = paramInfo_.xShape[DIM_ONE];
        shapeInfo.inShape[DIM_TWO] = paramInfo_.xShape[DIM_TWO] / paramInfo_.blockSize;
        shapeInfo.inShape[DIM_THREE] = paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FOUR] = paramInfo_.xShape[DIM_THREE] / paramInfo_.blockSize;
        shapeInfo.inShape[DIM_FIVE] = paramInfo_.blockSize;
        for (int64_t i = 0; i < DIM_SIX; i++) {
            shapeInfo.perm[i] = nchwPerm_[i];
            shapeInfo.outShape[i] = shapeInfo.inShape[shapeInfo.perm[i]];
        }
    }
}

ge::graphStatus SpaceToDepthTilingForAscendC(gert::TilingContext* context,
                                             const TransposeCompilerInfo* transposeCompileInfo)
{
    OP_LOGD(context->GetNodeName(), "Start SpaceToDepthTilingForAscendC.");
    SpaceToDepthTiling tilingObject(context);

    OP_CHECK_IF(tilingObject.ParametersVerifying() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SpaceToDepthTiling failed to verify params!"),
                return ge::GRAPH_FAILED);

    // construct an equivalent Transpose inputShapeInfo
    ShapeInfo inputShapeInfo;
    tilingObject.ProcessShapeInfo(inputShapeInfo);

    SpaceToDepthTilingData tilingData;
    SpaceToDepthCompileInfo compileInfo;
    compileInfo.transposeCompilerInfo.coreNum = transposeCompileInfo->coreNum;
    compileInfo.transposeCompilerInfo.ubSize = transposeCompileInfo->ubSize;

    TransposeNddmaTiling transposeTilingObject(context);
    OP_CHECK_IF((transposeTilingObject.TilingForReleatedTranspose(context, &tilingData.transposeOpTiling,
                                                                  &compileInfo.transposeCompilerInfo,
                                                                  inputShapeInfo) == ge::GRAPH_FAILED),
                OP_LOGE(context->GetNodeName(), "Transpose Tiling failed"), return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context->GetNodeName(), "SpaceToDepthTilingForAscendC success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForRelatedToTranspose(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("SpaceToDepth", "SpaceToDepth context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepareForRelatedToTranspose.");
    TilingPrepareTransposeForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForSpaceToDepth(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingForSpaceToDepth");
    auto compile_info = context->GetCompileInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    return SpaceToDepthTilingForAscendC(context, compile_info);
}

IMPL_OP_OPTILING(SpaceToDepth)
    .Tiling(TilingForSpaceToDepth)
    .TilingParse<TransposeCompilerInfo>(TilingPrepareForRelatedToTranspose);
} // namespace optiling
