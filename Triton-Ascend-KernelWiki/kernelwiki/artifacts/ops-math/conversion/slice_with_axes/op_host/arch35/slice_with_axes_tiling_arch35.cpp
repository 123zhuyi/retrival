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
#include "conversion/slice_with_axes/op_kernel/arch35/slice_with_axes_tiling_data.h"
#include "conversion/slice_with_axes/op_kernel/arch35/slice_with_axes_tiling_key.h"
#include "platform/platform_ascendc.h"
#include "op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"

namespace optiling {

static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_OFFSETS = 1;
static constexpr size_t INPUT_IDX_SIZE = 2;
static constexpr uint32_t MAX_BUFFER_SIZE = 64 * 1024;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr double MIN_CORE_UTIL_RATIO = 0.8;

class SliceWithAxesTiling {
public:
    explicit SliceWithAxesTiling(gert::TilingContext* context) : context_(context)
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

    int64_t inShape_[MAX_AXIS_COUNT]{0};
    int64_t outShape_[MAX_AXIS_COUNT]{0};
    int64_t fullOffsets_[MAX_AXIS_COUNT]{0};
    uint8_t rank_{0};
    int32_t dSize_{0};

    uint32_t bufferSize_{0};
    uint32_t bufferSizeElements_{0};
    uint32_t ubBlockElements_{0};
    uint32_t cacheLineElements_{0};

    uint8_t ubAxis_{0};
    uint32_t ubFactor_{0};
    uint64_t totalCount_{0};
    uint64_t perCoreCount_{0};
};

ge::graphStatus SliceWithAxesTiling::GetSocInfo()
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

ge::graphStatus SliceWithAxesTiling::GetParams()
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
    rank_ = static_cast<uint8_t>(xShape.GetDimNum());
    OP_CHECK_IF(
        rank_ == 0 || rank_ > MAX_AXIS_COUNT,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(rank_).c_str(), "The shape dim of x must be in [1, 8]."),
        return ge::GRAPH_FAILED);
    for (uint8_t i = 0; i < rank_; ++i) {
        inShape_[i] = xShape.GetDim(i);
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto axesPtr = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axesPtr);
    int64_t axesLen = axesPtr->GetSize();
    const int64_t* axesData = axesPtr->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, axesData);

    gert::Shape offsetsShape, sizeShape;
    bool hasOffsets = Ops::Base::GetConstIntToShape(context_, INPUT_IDX_OFFSETS, offsetsShape);
    bool hasSize = Ops::Base::GetConstIntToShape(context_, INPUT_IDX_SIZE, sizeShape);

    for (uint8_t i = 0; i < rank_; ++i) {
        fullOffsets_[i] = 0;
        outShape_[i] = inShape_[i];
    }

    if (hasOffsets && hasSize) {
        for (int64_t k = 0; k < axesLen; ++k) {
            int64_t ax = axesData[k];
            OP_CHECK_IF(
                ax < 0 || ax >= rank_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "axis", std::to_string(ax).c_str(),
                    "The value of axis must be in [0, rank)."),
                return ge::GRAPH_FAILED);
            fullOffsets_[ax] = offsetsShape[k];
            int64_t sz = sizeShape[k];
            if (sz == -1) {
                outShape_[ax] = inShape_[ax] - fullOffsets_[ax];
            } else {
                outShape_[ax] = sz;
            }
        }
    }

    for (uint8_t i = 0; i < rank_; ++i) {
        OP_CHECK_IF(
            outShape_[i] <= 0,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context_->GetNodeName(), "y", std::to_string(outShape_[i]).c_str(), "The shape dim of y must be >= 0."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

void SliceWithAxesTiling::InitElementSizes()
{
    bufferSize_ = std::min(ubSize_ / BUFFER_NUM, MAX_BUFFER_SIZE);
    bufferSizeElements_ = bufferSize_ / static_cast<uint32_t>(dSize_);
    ubBlockElements_ = ubBlockSize_ / static_cast<uint32_t>(dSize_);
    cacheLineElements_ = cacheLineSize_ / static_cast<uint32_t>(dSize_);
}

void SliceWithAxesTiling::DoCacheLineTiling(
    int64_t alignedLast, int32_t& startAxis, int64_t& clInnerProduct, int64_t& minUbFactor, int64_t& outerProduct)
{
    int32_t last = static_cast<int32_t>(rank_) - 1;
    startAxis = last;
    clInnerProduct = 1;
    for (int32_t ax = last; ax >= 0; --ax) {
        int64_t dimSize = (ax == last) ? alignedLast : outShape_[ax];
        if (dimSize * clInnerProduct >= static_cast<int64_t>(cacheLineElements_)) {
            startAxis = ax;
            break;
        }
        clInnerProduct *= dimSize;
    }

    int64_t startAxisStep = (startAxis == last) ? static_cast<int64_t>(ubBlockElements_) : 1;
    int64_t clMinFactor = Ops::Base::CeilDiv(static_cast<int64_t>(cacheLineElements_), clInnerProduct);
    minUbFactor = Ops::Base::CeilAlign(std::max(startAxisStep, clMinFactor), startAxisStep);

    outerProduct = 1;
    for (int32_t ax = 0; ax < startAxis; ++ax) {
        outerProduct *= outShape_[ax];
    }
}

void SliceWithAxesTiling::DoTilingSearch(
    int64_t alignedLast, int32_t startAxis, int64_t minUbFactor, int64_t clInnerProduct, int64_t outerProduct)
{
    int32_t last = static_cast<int32_t>(rank_) - 1;
    uint8_t bestAxis = 0;
    uint32_t bestFactor = 1;
    uint64_t bestTotalCount = 1;
    uint64_t bestRealCore = 0;

    int64_t cumInnerProduct = clInnerProduct;
    for (int32_t ax = startAxis; ax >= 0; --ax) {
        int64_t dimSize = (ax == last) ? alignedLast : outShape_[ax];
        int64_t step = (ax == last) ? static_cast<int64_t>(ubBlockElements_) : 1;
        int64_t factorStart = (ax == startAxis) ? minUbFactor : step;

        for (int64_t factor = factorStart; factor <= dimSize; factor += step) {
            int64_t blockSize = factor * cumInnerProduct;
            if (blockSize > static_cast<int64_t>(bufferSizeElements_)) {
                break;
            }
            uint64_t totalCount = static_cast<uint64_t>(outerProduct * Ops::Base::CeilDiv(dimSize, factor));
            uint64_t perCoreCount = Ops::Base::CeilDiv(totalCount, static_cast<uint64_t>(coreNum_));
            uint64_t realCoreNum = Ops::Base::CeilDiv(totalCount, perCoreCount);
            if (static_cast<double>(realCoreNum) >= static_cast<double>(coreNum_) * MIN_CORE_UTIL_RATIO ||
                realCoreNum > bestRealCore) {
                bestAxis = static_cast<uint8_t>(ax);
                bestFactor = static_cast<uint32_t>(factor);
                bestTotalCount = totalCount;
                if (realCoreNum > bestRealCore) {
                    bestRealCore = realCoreNum;
                }
            }
        }

        if (ax > 0) {
            outerProduct /= outShape_[ax - 1];
            cumInnerProduct *= dimSize;
        }
    }

    ubAxis_ = bestAxis;
    ubFactor_ = bestFactor;
    totalCount_ = bestTotalCount;
}

ge::graphStatus SliceWithAxesTiling::SetTilingData()
{
    perCoreCount_ = Ops::Base::CeilDiv(totalCount_, static_cast<uint64_t>(coreNum_));
    uint64_t realCoreNum = Ops::Base::CeilDiv(totalCount_, perCoreCount_);

    OP_LOGI(
        context_, "SliceWithAxes tiling: rank=%u ubAxis=%u ubFactor=%u totalCount=%lu perCoreCount=%lu realCoreNum=%lu",
        rank_, ubAxis_, ubFactor_, totalCount_, perCoreCount_, realCoreNum);

    auto tilingData = context_->GetTilingData<SliceWithAxesTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    for (size_t i = 0; i < MAX_AXIS_COUNT; ++i) {
        tilingData->inShape[i] = inShape_[i];
        tilingData->outShape[i] = outShape_[i];
        tilingData->fullOffsets[i] = fullOffsets_[i];
    }
    tilingData->totalCount = totalCount_;
    tilingData->perCoreCount = perCoreCount_;
    tilingData->ubAxis = ubAxis_;
    tilingData->ubFactor = ubFactor_;
    tilingData->bufferSize = bufferSize_;
    tilingData->rank = rank_;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(ubAxis_);
    OP_LOGI(context_, "tilingKey is %lu, ubAxis %u", tilingKey, ubAxis_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum);

    OP_LOGI(
        context_,
        "SliceWithAxes tilingData: rank=%u ubAxis=%u ubFactor=%u bufferSize=%u totalCount=%lu perCoreCount=%lu "
        "inShape=[%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld] outShape=[%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld] "
        "offsets=[%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld]",
        tilingData->rank, tilingData->ubAxis, tilingData->ubFactor, tilingData->bufferSize, tilingData->totalCount,
        tilingData->perCoreCount, tilingData->inShape[0], tilingData->inShape[1], tilingData->inShape[2],
        tilingData->inShape[3], tilingData->inShape[4], tilingData->inShape[5], tilingData->inShape[6],
        tilingData->inShape[7], tilingData->outShape[0], tilingData->outShape[1], tilingData->outShape[2],
        tilingData->outShape[3], tilingData->outShape[4], tilingData->outShape[5], tilingData->outShape[6],
        tilingData->outShape[7], tilingData->fullOffsets[0], tilingData->fullOffsets[1], tilingData->fullOffsets[2],
        tilingData->fullOffsets[3], tilingData->fullOffsets[4], tilingData->fullOffsets[5], tilingData->fullOffsets[6],
        tilingData->fullOffsets[7]);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SliceWithAxesTiling::ComputeAndSetTiling()
{
    InitElementSizes();

    int32_t last = static_cast<int32_t>(rank_) - 1;
    int64_t alignedLast = Ops::Base::CeilAlign(outShape_[last], static_cast<int64_t>(ubBlockElements_));
    int32_t startAxis;
    int64_t clInnerProduct, minUbFactor, outerProduct;
    DoCacheLineTiling(alignedLast, startAxis, clInnerProduct, minUbFactor, outerProduct);

    DoTilingSearch(alignedLast, startAxis, minUbFactor, clInnerProduct, outerProduct);
    return SetTilingData();
}

ge::graphStatus SliceWithAxesTiling::DoTiling()
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

static ge::graphStatus SliceWithAxesTilingFunc(gert::TilingContext* context)
{
    SliceWithAxesTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingParseForSliceWithAxes([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SliceWithAxes)
    .Tiling(SliceWithAxesTilingFunc)
    .TilingInputsDataDependency({INPUT_IDX_OFFSETS, INPUT_IDX_SIZE})
    .TilingParse<SliceWithAxesCompileInfo>(TilingParseForSliceWithAxes);
} // namespace optiling
