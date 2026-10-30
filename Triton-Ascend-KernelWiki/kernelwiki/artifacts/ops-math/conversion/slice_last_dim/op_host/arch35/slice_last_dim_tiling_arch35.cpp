/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "conversion/slice_last_dim/op_kernel/arch35/slice_last_dim_tiling_data.h"
#include "conversion/slice_last_dim/op_kernel/arch35/slice_last_dim_tiling_key.h"
#include "platform/platform_ascendc.h"
#include "op_host/util/platform_util.h"
#include "op_host/util/math_util.h"

namespace optiling {

static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t MAX_RANK = 8;
static constexpr uint32_t MAX_BUFFER_SIZE = 64 * 1024;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr double MIN_CORE_UTIL_RATIO = 0.8;

class SliceLastDimTiling {
public:
    explicit SliceLastDimTiling(gert::TilingContext* context) : context_(context)
    {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetSocInfo();
    ge::graphStatus GetParams();
    ge::graphStatus ComputeAndSetTiling();

    void InitElementSizes();
    void DoCacheLineTiling(
        int64_t alignedLast, int32_t& startAxis, int64_t& clInnerProduct, int64_t& minUbFactor, int64_t& outerProduct);
    void DoTilingSearch(
        int64_t alignedLast, int32_t startAxis, int64_t minUbFactor, int64_t clInnerProduct, int64_t outerProduct);
    ge::graphStatus SetTilingData();

    gert::TilingContext* context_;
    uint32_t ubSize_{0};
    uint32_t coreNum_{0};
    uint32_t ubBlockSize_{0};
    uint32_t cacheLineSize_{0};

    int64_t inShape_[MAX_RANK]{0};
    int64_t rank_{0};
    int64_t start_{0};
    int64_t end_{0};
    int64_t stride_{1};
    int32_t dSize_{0};

    int64_t outerSize_{0};
    int64_t lastDimIn_{0};
    int64_t lastDimOut_{0};

    uint32_t bufferSize_{0};
    uint32_t bufferSizeElements_{0};
    uint32_t ubBlockElements_{0};
    uint32_t cacheLineElements_{0};

    uint8_t ubAxis_{0};
    uint32_t ubFactor_{0};
    uint64_t totalCount_{0};
    uint64_t perCoreCount_{0};
    uint8_t copyMode_{0};
};

ge::graphStatus SliceLastDimTiling::GetSocInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (coreNum_ == 0U),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "coreNum", std::to_string(coreNum_).c_str(), "The core num must be positive."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        (ubSize == 0U),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
            "Failed to get ub size, ub size must be positive."),
        return ge::GRAPH_FAILED);
    ubSize_ = static_cast<uint32_t>(ubSize);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF(
        (cacheLineSize_ == 0U),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "cacheLineSize", std::to_string(cacheLineSize_).c_str(),
            "Failed to get cache line size, cache line size must be positive."),
        return ge::GRAPH_FAILED);
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF(
        (ubBlockSize_ == 0U),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "ubBlockSize", std::to_string(ubBlockSize_).c_str(),
            "Failed to get block size, block size must be positive."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SliceLastDimTiling::GetParams()
{
    auto inputValueDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
    auto inputDataType = inputValueDesc->GetDataType();
    dSize_ = ge::GetSizeByDataType(inputDataType);
    OP_CHECK_IF(
        dSize_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "dSize", std::to_string(dSize_).c_str(), "The data type size must be positive."),
        return ge::GRAPH_FAILED);

    auto xInputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    auto xShape = xInputShape->GetStorageShape();
    rank_ = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(
        rank_ < 1 || rank_ > static_cast<int64_t>(MAX_RANK),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(rank_).c_str(), "The shape dim of x must be in [1, 8]."),
        return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < rank_; ++i) {
        inShape_[i] = xShape.GetDim(i);
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto startPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, startPtr);
    start_ = *startPtr;

    auto endPtr = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, endPtr);
    end_ = *endPtr;

    auto stridePtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr != nullptr) {
        stride_ = *stridePtr;
    } else {
        stride_ = 1;
    }
    OP_CHECK_IF(
        stride_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "stride", std::to_string(stride_).c_str(), "The value of stride must be >= 1."),
        return ge::GRAPH_FAILED);

    lastDimIn_ = inShape_[rank_ - 1];
    if (start_ < 0) {
        start_ += lastDimIn_;
    }
    if (end_ < 0) {
        end_ += lastDimIn_;
    }
    if (start_ < 0) {
        start_ = 0;
    }
    if (end_ > lastDimIn_) {
        end_ = lastDimIn_;
    }

    lastDimOut_ = Ops::Base::CeilDiv(end_ - start_, stride_);
    if (lastDimOut_ < 0) {
        lastDimOut_ = 0;
    }

    outerSize_ = 1;
    for (int64_t i = 0; i < rank_ - 1; ++i) {
        outerSize_ *= inShape_[i];
    }

    copyMode_ = (stride_ == 1) ? 0 : 1;

    return ge::GRAPH_SUCCESS;
}

void SliceLastDimTiling::InitElementSizes()
{
    bufferSize_ = std::min(ubSize_ / BUFFER_NUM, MAX_BUFFER_SIZE);
    bufferSizeElements_ = bufferSize_ / static_cast<uint32_t>(dSize_);
    ubBlockElements_ = ubBlockSize_ / static_cast<uint32_t>(dSize_);
    cacheLineElements_ = cacheLineSize_ / static_cast<uint32_t>(dSize_);
}

void SliceLastDimTiling::DoCacheLineTiling(
    int64_t alignedLast, int32_t& startAxis, int64_t& clInnerProduct, int64_t& minUbFactor, int64_t& outerProduct)
{
    int64_t outShape[2];
    outShape[0] = outerSize_;
    outShape[1] = lastDimOut_;

    startAxis = 1;
    clInnerProduct = 1;
    for (int32_t ax = 1; ax >= 0; --ax) {
        int64_t dimSize = (ax == 1) ? alignedLast : outShape[ax];
        if (dimSize * clInnerProduct >= static_cast<int64_t>(cacheLineElements_)) {
            startAxis = ax;
            break;
        }
        clInnerProduct *= dimSize;
    }

    int64_t startAxisStep = (startAxis == 1) ? static_cast<int64_t>(ubBlockElements_) : 1;
    int64_t clMinFactor = Ops::Base::CeilDiv(static_cast<int64_t>(cacheLineElements_), clInnerProduct);
    minUbFactor = Ops::Base::CeilAlign(std::max(startAxisStep, clMinFactor), startAxisStep);

    outerProduct = 1;
    for (int32_t ax = 0; ax < startAxis; ++ax) {
        outerProduct *= outShape[ax];
    }
}

void SliceLastDimTiling::DoTilingSearch(
    int64_t alignedLast, int32_t startAxis, int64_t minUbFactor, int64_t clInnerProduct, int64_t outerProduct)
{
    int64_t outShape[2];
    outShape[0] = outerSize_;
    outShape[1] = lastDimOut_;

    uint8_t bestAxis = 0;
    uint32_t bestFactor = 1;
    uint64_t bestTotalCount = 1;
    uint64_t bestRealCore = 0;

    int64_t cumInnerProduct = clInnerProduct;
    for (int32_t ax = startAxis; ax >= 0; --ax) {
        int64_t dimSize = (ax == 1) ? alignedLast : outShape[ax];
        int64_t step = (ax == 1) ? static_cast<int64_t>(ubBlockElements_) : 1;
        int64_t factorStart = (ax == startAxis) ? minUbFactor : step;

        for (int64_t factor = factorStart; factor <= dimSize; factor += step) {
            int64_t blockSize = factor * cumInnerProduct;
            if (blockSize > static_cast<int64_t>(bufferSizeElements_)) {
                break;
            }
            uint64_t tc = static_cast<uint64_t>(outerProduct * Ops::Base::CeilDiv(dimSize, factor));
            uint64_t pcc = Ops::Base::CeilDiv(tc, static_cast<uint64_t>(coreNum_));
            uint64_t rcn = Ops::Base::CeilDiv(tc, pcc);

            if (static_cast<double>(rcn) >= static_cast<double>(coreNum_) * MIN_CORE_UTIL_RATIO || rcn > bestRealCore) {
                bestAxis = static_cast<uint8_t>(ax);
                bestFactor = static_cast<uint32_t>(factor);
                bestTotalCount = tc;
                if (rcn > bestRealCore) {
                    bestRealCore = rcn;
                }
            }
        }

        if (ax > 0) {
            outerProduct /= outShape[ax - 1];
            cumInnerProduct *= dimSize;
        }
    }

    ubAxis_ = bestAxis;
    ubFactor_ = bestFactor;
    totalCount_ = bestTotalCount;
}

ge::graphStatus SliceLastDimTiling::SetTilingData()
{
    perCoreCount_ = Ops::Base::CeilDiv(totalCount_, static_cast<uint64_t>(coreNum_));
    uint64_t realCoreNum = Ops::Base::CeilDiv(totalCount_, perCoreCount_);

    OP_LOGI(
        context_,
        "SliceLastDim tiling: copyMode=%u ubAxis=%u ubFactor=%u totalCount=%lu perCoreCount=%lu realCoreNum=%lu",
        copyMode_, ubAxis_, ubFactor_, totalCount_, perCoreCount_, realCoreNum);

    auto tilingData = context_->GetTilingData<SliceLastDimTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->outerSize = outerSize_;
    tilingData->lastDimIn = lastDimIn_;
    tilingData->lastDimOut = lastDimOut_;
    tilingData->start = start_;
    tilingData->stride = stride_;
    tilingData->totalCount = totalCount_;
    tilingData->perCoreCount = perCoreCount_;
    tilingData->ubFactor = ubFactor_;
    tilingData->bufferSize = bufferSize_;

    context_->SetTilingKey(GET_TPL_TILING_KEY(copyMode_, ubAxis_));
    context_->SetBlockDim(realCoreNum);

    OP_LOGI(
        context_,
        "SliceLastDim tilingData: outerSize=%ld lastDimIn=%ld lastDimOut=%ld start=%ld stride=%ld "
        "totalCount=%lu perCoreCount=%lu ubFactor=%u bufferSize=%u",
        tilingData->outerSize, tilingData->lastDimIn, tilingData->lastDimOut, tilingData->start, tilingData->stride,
        tilingData->totalCount, tilingData->perCoreCount, tilingData->ubFactor, tilingData->bufferSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SliceLastDimTiling::ComputeAndSetTiling()
{
    InitElementSizes();

    int64_t alignedLast = Ops::Base::CeilAlign(lastDimOut_, static_cast<int64_t>(ubBlockElements_));
    int32_t startAxis;
    int64_t clInnerProduct, minUbFactor, outerProduct;
    DoCacheLineTiling(alignedLast, startAxis, clInnerProduct, minUbFactor, outerProduct);

    DoTilingSearch(alignedLast, startAxis, minUbFactor, clInnerProduct, outerProduct);
    return SetTilingData();
}

ge::graphStatus SliceLastDimTiling::DoTiling()
{
    auto ret = GetSocInfo();
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "GetSocInfo", "failed", ""), return ret);
    ret = GetParams();
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "GetParams", "failed", ""), return ret);
    ret = ComputeAndSetTiling();
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ComputeAndSetTiling", "failed", ""),
        return ret);

    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SliceLastDimTilingFunc(gert::TilingContext* context)
{
    SliceLastDimTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingParseForSliceLastDim([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SliceLastDim)
    .Tiling(SliceLastDimTilingFunc)
    .TilingParse<SliceLastDimCompileInfo>(TilingParseForSliceLastDim);
} // namespace optiling
