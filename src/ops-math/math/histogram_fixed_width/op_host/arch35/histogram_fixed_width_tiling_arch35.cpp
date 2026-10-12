/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "math/histogram_fixed_width/op_host/arch35/histogram_fixed_width_tiling_arch35.h"

namespace optiling {

ge::graphStatus HistogramFixedWidthTiling::GetSocInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNum_ == 0U),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "core num",
                                                      std::to_string(coreNum_).c_str(), "must be greater than 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF((ubSize_ == 0U),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ub size",
                                                      std::to_string(ubSize_).c_str(), "must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramFixedWidthTiling::ReadRangeMinMax(const ge::DataType xDtype, float& minVal, float& maxVal)
{
    auto rangeTensor = context_->GetInputTensor(HFW_INPUT_IDX_RANGE);
    switch (xDtype) {
        case ge::DT_FLOAT: {
            const float* rangeData = rangeTensor->GetData<float>();
            if (rangeData == nullptr) {
                return ge::GRAPH_NULL_PTR;
            }
            minVal = rangeData[0];
            maxVal = rangeData[1];
            break;
        }
        case ge::DT_FLOAT16: {
            const uint16_t* rangeData = rangeTensor->GetData<uint16_t>();
            if (rangeData == nullptr) {
                return ge::GRAPH_NULL_PTR;
            }
            minVal = static_cast<float>(Ops::Base::fp16_t(rangeData[0]));
            maxVal = static_cast<float>(Ops::Base::fp16_t(rangeData[1]));
            break;
        }
        case ge::DT_INT32: {
            const int32_t* rangeData = rangeTensor->GetData<int32_t>();
            if (rangeData == nullptr) {
                return ge::GRAPH_NULL_PTR;
            }
            minVal = static_cast<float>(rangeData[0]);
            maxVal = static_cast<float>(rangeData[1]);
            break;
        }
        case ge::DT_INT64: {
            const int64_t* rangeData = rangeTensor->GetData<int64_t>();
            if (rangeData == nullptr) {
                return ge::GRAPH_NULL_PTR;
            }
            minVal = static_cast<float>(rangeData[0]);
            maxVal = static_cast<float>(rangeData[1]);
            break;
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "range", Ops::Base::ToString(xDtype).c_str(),
                "The dtype of range must be within the range DT_FLOAT, DT_FLOAT16, DT_INT32 and DT_INT64");
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramFixedWidthTiling::ValidateRange(const ge::DataType xDtype)
{
    auto rangeDesc = context_->GetInputDesc(HFW_INPUT_IDX_RANGE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rangeDesc);
    if (rangeDesc->GetDataType() != xDtype) {
        std::string errorMsg = "The dtype of range must be the same as " + Ops::Base::ToString(xDtype) + " of x";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "range",
                                              Ops::Base::ToString(rangeDesc->GetDataType()).c_str(), errorMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    auto rangeShape = context_->GetInputShape(HFW_INPUT_IDX_RANGE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rangeShape);
    auto rangeLength = rangeShape->GetStorageShape().GetShapeSize();
    if (rangeLength != HFW_RANGE_LENGTH) {
        std::string errorMsg = "The shape size of range should be equal to " + std::to_string(HFW_RANGE_LENGTH);
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "range", std::to_string(rangeLength).c_str(),
                                                  errorMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    float minVal = 0.0f;
    float maxVal = 0.0f;
    auto ret = ReadRangeMinMax(xDtype, minVal, maxVal);
    if (ret != ge::GRAPH_SUCCESS && ret != ge::GRAPH_NULL_PTR) {
        return ret;
    }
    if (ret == ge::GRAPH_SUCCESS) {
        OP_CHECK_IF(
            minVal >= maxVal,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "range",
                                                  (std::to_string(minVal) + " and " + std::to_string(maxVal)).c_str(),
                                                  "The value of max must be greater than min"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramFixedWidthTiling::ParamCheck()
{
    auto inputShape = context_->GetInputShape(HFW_INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    totalLength_ = inputShape->GetStorageShape().GetShapeSize();

    auto inputDesc = context_->GetInputDesc(HFW_INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType xDtype = inputDesc->GetDataType();

    if (xDtype != ge::DT_FLOAT && xDtype != ge::DT_INT32 && xDtype != ge::DT_INT64 && xDtype != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
            "The dtype of x must be within the range DT_FLOAT, DT_FLOAT16, DT_INT32 and DT_INT64");
        return ge::GRAPH_FAILED;
    }

    auto nbinsTensor = context_->GetInputTensor(HFW_INPUT_IDX_NBINS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, nbinsTensor);
    const int32_t* nbinsPtr = nbinsTensor->GetData<int32_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, nbinsPtr);
    bins_ = static_cast<int64_t>(*nbinsPtr);

    auto outputShape = context_->GetOutputShape(HFW_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outputDataLength = outputShape->GetStorageShape().GetShapeSize();
    if (outputDataLength != bins_) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "y and nbins",
                                                   std::to_string(outputDataLength).c_str(),
                                                   "The shape sizes of y and nbins must be the same");
        return ge::GRAPH_FAILED;
    }

    return ValidateRange(xDtype);
}

ge::graphStatus HistogramFixedWidthTiling::CalcTiling()
{
    uint64_t ubSizeAvail = ubSize_ - HFW_SIMT_DCACHE_SIZE;
    int64_t ubNumCanUse = static_cast<int64_t>(ubSizeAvail / HFW_SIZE_OF_INT32);
    int64_t ubLoopNum = Ops::Base::CeilDiv(bins_, ubNumCanUse);

    if (bins_ < ubNumCanUse) {
        loadMode_ = HFW_TPL_LOAD_MODE_UB_FULL;
        context_->SetLocalMemorySize(ubSizeAvail);
    } else if (totalLength_ > bins_ / HFW_GM_ATOMIC_ADD_FACTOR) {
        loadMode_ = HFW_TPL_LOAD_MODE_UB_NOT_FULL;
        context_->SetLocalMemorySize(ubSizeAvail);
    } else {
        loadMode_ = HFW_TPL_LOAD_MODE_UB_NOT_FULL_SIMT;
    }

    int64_t formerLength = Ops::Base::CeilDiv(totalLength_, static_cast<int64_t>(coreNum_));
    int64_t needXCoreNum = Ops::Base::CeilDiv(totalLength_, formerLength);
    int64_t tailLength = totalLength_ - (needXCoreNum - 1) * formerLength;

    int64_t clearYFactor = Ops::Base::CeilDiv(bins_, static_cast<int64_t>(coreNum_));
    OP_CHECK_IF((clearYFactor == 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "clearYFactor",
                                                      std::to_string(clearYFactor).c_str(), "must be greater than 0"),
                return ge::GRAPH_FAILED);
    int64_t clearYCoreNum = Ops::Base::CeilDiv(bins_, clearYFactor);
    int64_t clearYTail = bins_ - (clearYCoreNum - 1) * clearYFactor;
    int64_t needCoreNum = std::max(needXCoreNum, clearYCoreNum);

    auto tilingData = context_->GetTilingData<HistogramFixedWidthSimtTilingData>();
    tilingData->bins = static_cast<int32_t>(bins_);
    tilingData->ubNumCanUse = static_cast<uint32_t>(ubNumCanUse);
    tilingData->ubLoopNum = static_cast<uint32_t>(ubLoopNum);
    tilingData->needXCoreNum = static_cast<uint32_t>(needXCoreNum);
    tilingData->formerLength = formerLength;
    tilingData->tailLength = tailLength;
    tilingData->clearYCoreNum = static_cast<uint32_t>(clearYCoreNum);
    tilingData->clearYFactor = clearYFactor;
    tilingData->clearYTail = clearYTail;
    tilingData->needCoreNum = static_cast<uint32_t>(needCoreNum);

    uint64_t tilingKey = GET_TPL_TILING_KEY(loadMode_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    context_->SetScheduleMode(1);

    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;

    OP_LOGD(context_->GetNodeName(), "loadMode=%lu, needCoreNum=%ld", loadMode_, needCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramFixedWidthTiling::DoTiling()
{
    if (ParamCheck() == ge::GRAPH_FAILED || GetSocInfo() == ge::GRAPH_FAILED || CalcTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4HistogramFixedWidth(gert::TilingContext* context)
{
    HistogramFixedWidthTiling tiling{context};
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForHistogramFixedWidth([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HistogramFixedWidth)
    .Tiling(Tiling4HistogramFixedWidth)
    .TilingInputsDataDependency({HFW_INPUT_IDX_RANGE, HFW_INPUT_IDX_NBINS})
    .TilingParse<HistogramFixedWidthCompileInfo>(TilingPrepareForHistogramFixedWidth);
} // namespace optiling
