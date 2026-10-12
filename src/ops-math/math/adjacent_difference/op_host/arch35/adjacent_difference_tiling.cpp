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
 * \file adjacent_difference_tiling.cpp
 * \brief
 */
#include <vector>
#include "tiling/tiling_api.h"
#include "adjacent_difference_tiling.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/platform_util.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling
{
const std::string OP_NAME = "AdjacentDifference";
static constexpr size_t WORKSPACE_SIZE = 1;
static ge::graphStatus TilingPrepare4AdjacentDifference([[maybe_unused]] gert::TilingParseContext* context) {
    OP_LOGI("AdjacentDifference", "TilingPrepare4AdjacentDifference running.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AdjacentDifference(gert::TilingContext* context) {
    OP_LOGI("AdjacentDifference", "Begin to do Tiling4AdjacentDifference.");
    OP_CHECK_IF(nullptr == context, OP_LOGE("AdjacentDifference", "TilingContext is null, failed."),
        return ge::GRAPH_FAILED);
    AdjacentDifferenceTilingData tiling;

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    ge::DataType outputType = outputDesc->GetDataType();
    int64_t totalSize = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto extrSize = sizeof(int32_t);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto yDtypePtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDtypePtr);
    int64_t yDtype = *yDtypePtr;
    if (static_cast<int64_t>(outputType) != yDtype) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(outputType)) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(yDtype));
        std::string reasonMsg = "The dtype of y and attr parameter y_dtype must be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "AdjacentDifference", "y and y_dtype", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (yDtype == static_cast<int64_t>(ge::DataType::DT_INT64)) {
        OP_LOGI("AdjacentDifference", "yDtype is DT_INT64");
        extrSize += sizeof(int32_t);
    } else if (yDtype == static_cast<int64_t>(ge::DataType::DT_INT32)) {
        OP_LOGI("AdjacentDifference", "yDtype is DT_INT32");
    } else {
        OP_LOGE_FOR_INVALID_VALUE(
            "AdjacentDifference", "y_dtype",
            ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(yDtype)).c_str(), "int32 or int64");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatform;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    int64_t ubSize = static_cast<int64_t>(ubSizePlatform);
    int64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    auto sizeOfInputType = GetSizeByDataType(dataType);
    OP_CHECK_IF(sizeOfInputType == 0,
        OP_LOGE_FOR_INVALID_DTYPE("AdjacentDifference", "x",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str(), "non zero"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(aivCoreNum == 0, OP_LOGE("AdjacentDifference", "aivCoreNum is zero."), return ge::GRAPH_FAILED);
    int64_t blockNum = Ops::Base::GetUbBlockSize(context) / sizeOfInputType;
    OP_CHECK_IF(blockNum == 0, OP_LOGE("AdjacentDifference", "blockNum is zero."), return ge::GRAPH_FAILED);
    int64_t tilingNum = std::min(
        std::max(totalSize / aivCoreNum, static_cast<int64_t>(Ops::Base::GetVRegSize(context) / sizeOfInputType)),
        static_cast<int64_t>(ubSize / (sizeOfInputType + extrSize)));
    tilingNum = tilingNum / blockNum * blockNum;

    tiling.set_totalSize(totalSize);
    tiling.set_tilingNum(tilingNum);
    tiling.set_coreNum(aivCoreNum);

    // set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSPACE_SIZE;
    context->SetTilingKey(1);
    context->SetBlockDim(aivCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdjacentDifference)
    .Tiling(Tiling4AdjacentDifference)
    .TilingParse<AdjacentDifferenceCompileInfo>(TilingPrepare4AdjacentDifference);
}  // namespace optiling