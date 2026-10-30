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
 * \file eye_tiling.cc
 * \brief
 */
#include "eye_tiling_arch35.h"
#include <vector>
#include "log/log.h"
#include "op_host/util/math_util.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
using namespace Ops::Base;
constexpr int64_t OUTPUT_Y_IDX = 0;
constexpr int64_t ATTR_NUM_ROWS_IDX = 0;
constexpr int64_t ATTR_NUM_COLUMNS_IDX = 1;
constexpr int64_t ATTR_BATCH_SHAPE_IDX = 2;
constexpr int64_t ATTR_DTYPE_IDX = 3;
constexpr int64_t Y_RIGHT_ROW_IDX = 2;
constexpr size_t BATCH_MAX_DIM = 6;

constexpr int64_t SIMD_TILING_KEY = 1000;
constexpr int64_t LARGE_SHAPE_TILING_KEY = 2000;
constexpr int64_t LARGE_AXIS_TAG = 100;

constexpr int64_t MAX_UINT32_NUM = 4294967295;
constexpr int64_t TMP_SIMD_CONDITION = 32796; // 32K
constexpr int64_t USE_UB_MAX_SIZE = 65536; // 64K
constexpr int64_t UB_PROCESS_MAX_NUM = 65536;
constexpr int64_t RESERVED_UB_SIZE = 8192; // 8K
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216; // 16M
constexpr int64_t V_REG_SIZE = 256U;

const std::map<ge::DataType, int64_t> EYE_SUPPORTED_DTYPE = {
    { ge::DT_FLOAT, 0 }, { ge::DT_FLOAT16, 1 }, { ge::DT_UINT8, 2 }, { ge::DT_INT8, 3 }, { ge::DT_INT32, 4 },
    { ge::DT_INT16, 5 }, { ge::DT_BOOL, 6 },    { ge::DT_INT64, 7 }, { ge::DT_BF16, 8 }
};

bool EyeTiling::IsCapable()
{
    return true;
}

ge::graphStatus EyeTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const EyeCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    ubSize_ -= RESERVED_UB_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EyeTiling::GetShapeAttrsInfo()
{
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto numRows = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ROWS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numRows);
    numRows_ = static_cast<int64_t>(*numRows);
    OP_CHECK_IF((numRows_ <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_rows",
            std::to_string(numRows_).c_str(),
            "The value of num_rows must be greater than 0."),
        return ge::GRAPH_FAILED);

    auto numColumns = attrs->GetAttrPointer<int64_t>(ATTR_NUM_COLUMNS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numColumns);
    numColumns_ = static_cast<int64_t>(*numColumns);
    if (numColumns_ <= 0) {
        numColumns_ = numRows_;
    }

    auto outputYPtr = context_->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    dType_ = outputYPtr->GetDataType();
    bool dtypeInValid = EYE_SUPPORTED_DTYPE.find(dType_) == EYE_SUPPORTED_DTYPE.end();
    OP_CHECK_IF(dtypeInValid,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y",
            Ops::Base::ToString(dType_).c_str(),
            "The dtype of y must be within the range [DT_FLOAT, DT_FLOAT16, DT_UINT8, DT_INT8, DT_INT32, DT_INT16, DT_BOOL, DT_INT64, DT_BF16]."),
        return ge::GRAPH_FAILED);

    return CheckOutputShape();
}

ge::graphStatus EyeTiling::CheckOutputShape()
{
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto batchShapePtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_BATCH_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, batchShapePtr);
    auto batchShapeArray = reinterpret_cast<const int64_t *>(batchShapePtr->GetData());
    size_t batchDim = batchShapePtr->GetSize();
    OP_CHECK_IF((batchDim > BATCH_MAX_DIM),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "batch_shape",
            std::to_string(batchDim).c_str(),
            ("The shape dim of batch_shape must be within the range [0, " + std::to_string(BATCH_MAX_DIM) + "].").c_str()),
        return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    size_t yDimNum = yShape.GetDimNum();
    size_t yDimExp = batchDim + Y_RIGHT_ROW_IDX;
    OP_CHECK_IF((yDimNum != yDimExp),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "y",
            std::to_string(yDimNum).c_str(),
            ("The shape dim of y must be equal to " + std::to_string(yDimExp) + " .").c_str()),
        return ge::GRAPH_FAILED);

auto yRow = yShape.GetDim(yDimNum - Y_RIGHT_ROW_IDX);
    auto yCol = yShape.GetDim(yDimNum - 1);
    OP_CHECK_IF((yRow != numRows_),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y",
            std::to_string(yRow).c_str(),
            ("The second-to-last dimension of y must be equal to " + std::to_string(numRows_) + ".").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((yCol != numColumns_),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y",
            std::to_string(yCol).c_str(),
            ("The last dimension of y must be equal to " + std::to_string(numColumns_) + ".").c_str()),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < batchDim; ++i) {
        OP_CHECK_IF((batchShapeArray[i] <= 0),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "batch_shape",
                std::to_string(batchShapeArray[i]).c_str(),
                "The value of batch_shape must be greater than 0."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((batchShapeArray[i] != yShape.GetDim(i)),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y",
                std::to_string(yShape.GetDim(i)).c_str(),
                ("Shape [" + std::to_string(i) + "] of y must be equal to " + std::to_string(batchShapeArray[i]) + ".").c_str()),
            return ge::GRAPH_FAILED);
        batch_ *= batchShapeArray[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EyeTiling::DoOpTiling()
{
    tilingData_.set_numRows(numRows_);
    tilingData_.set_numColumns(numColumns_);
    tilingData_.set_batch(batch_);

    int64_t numPerBatch = numRows_ * numColumns_;
    typeSize_ = ge::GetSizeByDataType(dType_);
    OP_CHECK_IF(typeSize_ <= 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y",
            Ops::Base::ToString(dType_).c_str(),
            "The dtype size of y must be greater than 0."),
        return ge::GRAPH_FAILED);
    if (numPerBatch * typeSize_ > TMP_SIMD_CONDITION) {
        allAxis_ = batch_ * numRows_ * numColumns_;
        loopLength_ = std::min(ubSize_, USE_UB_MAX_SIZE) / typeSize_;
        normBlockData_ = Ops::Base::CeilDiv(allAxis_, totalCoreNum_);
        usedCoreNum_ = Ops::Base::CeilDiv(allAxis_, normBlockData_);
        tailBlockData_ = allAxis_ - (usedCoreNum_ - 1) * normBlockData_;
        context_->SetScheduleMode(1);
    } else {
        int64_t vRegSize = V_REG_SIZE;
        int64_t ubProNum = std::min(std::min(ubSize_ - vRegSize, USE_UB_MAX_SIZE) / typeSize_, UB_PROCESS_MAX_NUM);
        loopLength_ = ubProNum / numPerBatch;
        normBlockData_ = Ops::Base::CeilDiv(batch_, totalCoreNum_);
        loopLength_ = std::min(loopLength_, normBlockData_);
        usedCoreNum_ = Ops::Base::CeilDiv(batch_, normBlockData_);
        tailBlockData_ = batch_ - (usedCoreNum_ - 1) * normBlockData_;
    }

    tilingData_.set_loopLength(loopLength_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_normBlockData(normBlockData_);
    tilingData_.set_tailBlockData(tailBlockData_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EyeTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t EyeTiling::GetTilingKey() const
{
    int64_t tilingKey = SIMD_TILING_KEY;
    if (numRows_ * numColumns_ * typeSize_ > TMP_SIMD_CONDITION) {
        tilingKey = LARGE_SHAPE_TILING_KEY;
        if (allAxis_ > MAX_UINT32_NUM) {
            tilingKey += LARGE_AXIS_TAG;
        }
    }
    return tilingKey + EYE_SUPPORTED_DTYPE.find(dType_)->second;
}

ge::graphStatus EyeTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EyeTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(usedCoreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void EyeTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "usedCoreNum: " << usedCoreNum_ << std::endl;
    info << "normBlockData: " << normBlockData_ << std::endl;
    info << "tailBlockData: " << tailBlockData_ << std::endl;
    info << "loopLength: " << loopLength_ << std::endl;
    info << "numRows:" << numRows_ << std::endl;
    info << "numColumns: " << numColumns_ << std::endl;
    info << "batch: " << batch_ << std::endl;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus TilingForEye(gert::TilingContext* context) {
    auto compile_info = reinterpret_cast<const EyeCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    EyeTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForEye(gert::TilingParseContext* context) {
    OP_LOGD(context->GetNodeName(), "TilingPrepareEyeForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<EyeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Eye op.
IMPL_OP_OPTILING(Eye).Tiling(TilingForEye).TilingParse<EyeCompileInfo>(TilingPrepareForEye);
} // namespace optiling
