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
#include "conversion/space_to_batch_nd/op_kernel/arch35/space_to_batch_nd_tiling_data.h"
#include "conversion/space_to_batch_nd/op_kernel/arch35/space_to_batch_nd_tiling_key.h"
#include "platform/platform_ascendc.h"
#include "op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"

namespace optiling {

static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_BS = 1;
static constexpr size_t INPUT_IDX_PADS = 2;
static constexpr uint32_t MAX_BUFFER_SIZE = 64 * 1024;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr double MIN_CORE_UTIL_RATIO = 0.8;

class SpaceToBatchNDTiling {
public:
    explicit SpaceToBatchNDTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetSocInfo();
    ge::graphStatus GetParams();
    ge::graphStatus ComputeShapes();
    ge::graphStatus ComputeAndSetTiling();

    void InitElementSizes();
    void DoCacheLineTiling(int64_t alignedInner, int32_t& startAxis, int64_t& clInnerProduct, int64_t& minUbFactor,
                           int64_t& outerProduct);
    void DoTiling(int64_t alignedInner, int32_t startAxis, int64_t minUbFactor, int64_t clInnerProduct,
                  int64_t outerProduct);
    ge::graphStatus SetTilingData();

    gert::TilingContext* context_;
    uint32_t ubSize_{0};
    uint32_t coreNum_{0};
    uint32_t ubBlockSize_{0};
    uint32_t cacheLineSize_{0};

    int64_t inShape_[MAX_RANK]{0};
    int64_t outShape_[MAX_RANK]{0};
    uint8_t rank_{0};
    uint8_t numSpatialDims_{0};
    int64_t blockShape_[MAX_SPATIAL]{0};
    int64_t padTop_[MAX_SPATIAL]{0};
    int64_t padBottom_[MAX_SPATIAL]{0};
    int64_t paddedShape_[MAX_SPATIAL]{0};
    int64_t batchProd_{0};
    int64_t innerSize_{0};
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

ge::graphStatus SpaceToBatchNDTiling::GetSocInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNum_ == 0U), OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF((ubSize == 0U), OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    ubSize_ = static_cast<uint32_t>(ubSize);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF((cacheLineSize_ == 0U), OP_LOGE(context_, "Failed to get cache line size."), return ge::GRAPH_FAILED);
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ == 0U), OP_LOGE(context_, "Failed to get ub block size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDTiling::GetParams()
{
    auto inputValueDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
    auto inputDataType = inputValueDesc->GetDataType();
    dSize_ = ge::GetSizeByDataType(inputDataType);
    OP_CHECK_IF(dSize_ <= 0, OP_LOGE(context_, "data size should be positive"), return ge::GRAPH_FAILED);

    auto xInputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    auto xShape = xInputShape->GetStorageShape();
    rank_ = xShape.GetDimNum();
    OP_CHECK_IF(rank_ < 2 || rank_ > MAX_RANK, OP_LOGE(context_, "invalid rank %u", rank_), return ge::GRAPH_FAILED);

    auto bsTensor = context_->GetInputTensor(INPUT_IDX_BS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bsTensor);
    numSpatialDims_ = bsTensor->GetShapeSize();
    OP_CHECK_IF(numSpatialDims_ == 0 || numSpatialDims_ > MAX_SPATIAL,
                OP_LOGE(context_, "invalid numSpatialDims %u", numSpatialDims_), return ge::GRAPH_FAILED);

    gert::Shape bsVec;
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context_, INPUT_IDX_BS, bsVec),
                OP_LOGE(context_, "get block_shape const data failed"), return ge::GRAPH_FAILED);

    gert::Shape padsVec;
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context_, INPUT_IDX_PADS, padsVec),
                OP_LOGE(context_, "get paddings const data failed"), return ge::GRAPH_FAILED);

    for (size_t i = 0; i < numSpatialDims_; i++) {
        blockShape_[i] = bsVec.GetDim(i);
        OP_CHECK_IF(blockShape_[i] <= 0, OP_LOGE(context_, "block_shape[%zu]=%ld must be > 0", i, blockShape_[i]),
                    return ge::GRAPH_FAILED);
        padTop_[i] = padsVec.GetDim(i * 2);
        padBottom_[i] = padsVec.GetDim(i * 2 + 1);
        OP_CHECK_IF(padTop_[i] < 0 || padBottom_[i] < 0, OP_LOGE(context_, "paddings must be non-negative"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDTiling::ComputeShapes()
{
    auto xInputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    auto xShape = xInputShape->GetStorageShape();

    int64_t batch = xShape.GetDim(0);
    batchProd_ = batch;
    for (size_t i = 0; i < numSpatialDims_; i++) {
        int64_t spatial = xShape.GetDim(i + 1);
        inShape_[i + 1] = spatial;
        paddedShape_[i] = spatial + padTop_[i] + padBottom_[i];
        OP_CHECK_IF(paddedShape_[i] % blockShape_[i] != 0,
                    OP_LOGE(context_, "padded spatial[%zu]=%ld not divisible by block_shape[%zu]=%ld", i,
                            paddedShape_[i], i, blockShape_[i]),
                    return ge::GRAPH_FAILED);
        batchProd_ *= blockShape_[i];
    }

    innerSize_ = 1;
    for (size_t i = numSpatialDims_ + 1; i < rank_; i++) {
        innerSize_ *= xShape.GetDim(i);
    }

    inShape_[0] = batch;
    inShape_[numSpatialDims_ + 1] = innerSize_;

    outShape_[0] = batchProd_;
    for (size_t i = 0; i < numSpatialDims_; i++) {
        outShape_[i + 1] = paddedShape_[i] / blockShape_[i];
    }
    outShape_[numSpatialDims_ + 1] = innerSize_;

    rank_ = numSpatialDims_ + 2;

    return ge::GRAPH_SUCCESS;
}

void SpaceToBatchNDTiling::InitElementSizes()
{
    bufferSize_ = std::min(ubSize_ / BUFFER_NUM, MAX_BUFFER_SIZE);
    bufferSizeElements_ = bufferSize_ / static_cast<uint32_t>(dSize_);
    ubBlockElements_ = ubBlockSize_ / static_cast<uint32_t>(dSize_);
    cacheLineElements_ = cacheLineSize_ / static_cast<uint32_t>(dSize_);
}

void SpaceToBatchNDTiling::DoCacheLineTiling(int64_t alignedInner, int32_t& startAxis, int64_t& clInnerProduct,
                                             int64_t& minUbFactor, int64_t& outerProduct)
{
    int32_t lastAxis = static_cast<int32_t>(rank_) - 1;
    startAxis = lastAxis;
    clInnerProduct = 1;
    for (int32_t ax = lastAxis; ax >= 0; --ax) {
        int64_t dimSize = (ax == lastAxis) ? alignedInner : outShape_[ax];
        if (dimSize * clInnerProduct >= static_cast<int64_t>(cacheLineElements_)) {
            startAxis = ax;
            break;
        }
        clInnerProduct *= dimSize;
    }

    int64_t startAxisStep = (startAxis == lastAxis) ? static_cast<int64_t>(ubBlockElements_) : 1;
    int64_t clMinFactor = Ops::Base::CeilDiv(static_cast<int64_t>(cacheLineElements_), clInnerProduct);
    minUbFactor = Ops::Base::CeilAlign(std::max(startAxisStep, clMinFactor), startAxisStep);

    outerProduct = 1;
    for (int32_t ax = 0; ax < startAxis; ++ax) {
        outerProduct *= outShape_[ax];
    }
}

void SpaceToBatchNDTiling::DoTiling(int64_t alignedInner, int32_t startAxis, int64_t minUbFactor,
                                    int64_t clInnerProduct, int64_t outerProduct)
{
    int32_t lastAxis = static_cast<int32_t>(rank_) - 1;
    uint8_t bestAxis = 0;
    uint32_t bestFactor = 1;
    uint64_t bestTotalCount = 1;
    uint64_t bestRealCore = 1;

    int64_t cumInnerProduct = clInnerProduct;
    for (int32_t ax = startAxis; ax >= 1; --ax) { // Only tile on spatial dimensions (ax >= 1)
        int64_t dimSize = (ax == lastAxis) ? alignedInner : outShape_[ax];
        int64_t step = (ax == lastAxis) ? static_cast<int64_t>(ubBlockElements_) : 1;
        int64_t factorStart = (ax == startAxis) ? minUbFactor : step;

        // 计算 ax 之后所有维度的乘积（统一按输出 shape 计算）
        int64_t innerProduct = 1;
        for (int32_t i = ax + 1; i < static_cast<int32_t>(rank_); ++i) {
            innerProduct *= outShape_[i];
        }

        for (int64_t factor = factorStart; factor <= dimSize; factor += step) {
            // 统一公式：cumInnerProduct 含 tail axis align, × dSize 即得每 factor 占用的字节数
            int64_t bufferBytesUsed = factor * cumInnerProduct * static_cast<int64_t>(dSize_);
            if (bufferBytesUsed > static_cast<int64_t>(bufferSize_)) {
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

        if (ax > 1) {
            outerProduct /= outShape_[ax - 1];
            cumInnerProduct *= dimSize;
        }
    }

    ubAxis_ = bestAxis;
    ubFactor_ = bestFactor;
    totalCount_ = bestTotalCount;
}

ge::graphStatus SpaceToBatchNDTiling::SetTilingData()
{
    perCoreCount_ = Ops::Base::CeilDiv(totalCount_, static_cast<uint64_t>(coreNum_));
    uint64_t realCoreNum = Ops::Base::CeilDiv(totalCount_, perCoreCount_);

    OP_LOGI(context_, "SpaceToBatchND tiling: ubAxis=%u ubFactor=%u totalCount=%lu perCoreCount=%lu realCoreNum=%lu",
            ubAxis_, ubFactor_, totalCount_, perCoreCount_, realCoreNum);

    auto tilingData = context_->GetTilingData<SpaceToBatchNDTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->rank = rank_;
    for (size_t i = 0; i < rank_; i++) {
        tilingData->inShape[i] = inShape_[i];
        tilingData->outShape[i] = outShape_[i];
    }
    tilingData->totalCount = totalCount_;
    tilingData->perCoreCount = perCoreCount_;
    tilingData->ubAxis = ubAxis_;
    tilingData->ubFactor = ubFactor_;
    tilingData->bufferSize = bufferSize_;
    tilingData->numSpatialDims = numSpatialDims_;
    for (size_t i = 0; i < numSpatialDims_; i++) {
        tilingData->blockShape[i] = blockShape_[i];
        tilingData->padTop[i] = padTop_[i];
        tilingData->padBottom[i] = padBottom_[i];
    }

    uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(rank_) - static_cast<uint64_t>(ubAxis_));

    OP_LOGI(context_, "tilingKey is %lu, ubAxis %u", tilingKey, ubAxis_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDTiling::ComputeAndSetTiling()
{
    InitElementSizes();

    int64_t alignedInner = Ops::Base::CeilAlign(outShape_[rank_ - 1], static_cast<int64_t>(ubBlockElements_));
    int32_t startAxis;
    int64_t clInnerProduct, minUbFactor, outerProduct;
    DoCacheLineTiling(alignedInner, startAxis, clInnerProduct, minUbFactor, outerProduct);

    DoTiling(alignedInner, startAxis, minUbFactor, clInnerProduct, outerProduct);
    return SetTilingData();
}

ge::graphStatus SpaceToBatchNDTiling::DoTiling()
{
    auto ret = GetSocInfo();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetSocInfo failed"), return ret);
    ret = GetParams();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetParams failed"), return ret);
    ret = ComputeShapes();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ComputeShapes failed"), return ret);
    ret = ComputeAndSetTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ComputeAndSetTiling failed"), return ret);

    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SpaceToBatchNDTilingFunc(gert::TilingContext* context)
{
    SpaceToBatchNDTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingParseForSpaceToBatchND([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SpaceToBatchND)
    .Tiling(SpaceToBatchNDTilingFunc)
    .TilingInputsDataDependency({INPUT_IDX_BS, INPUT_IDX_PADS})
    .TilingParse<SpaceToBatchNDCompileInfo>(TilingParseForSpaceToBatchND);
} // namespace optiling
