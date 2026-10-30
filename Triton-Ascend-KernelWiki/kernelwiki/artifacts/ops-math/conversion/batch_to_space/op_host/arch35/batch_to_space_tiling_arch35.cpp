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
#include "conversion/batch_to_space/op_kernel/arch35/batch_to_space_tiling_data.h"
#include "conversion/batch_to_space/op_kernel/arch35/batch_to_space_tiling_key.h"
#include "platform/platform_ascendc.h"
#include "op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"

namespace optiling {

static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_CROPS = 1;
static constexpr size_t CROPS_SIZE = 4;
static constexpr size_t AXIS_N = 0;
static constexpr size_t AXIS_H = 1;
static constexpr size_t AXIS_W = 2;
static constexpr size_t AXIS_C = 3;
static constexpr uint32_t MAX_BUFFER_SIZE = 64 * 1024;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr double MIN_CORE_UTIL_RATIO = 0.8;

class BatchToSpaceTiling {
public:
    explicit BatchToSpaceTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetSocInfo();
    ge::graphStatus GetParams();
    ge::graphStatus ComputeAndSetTiling();

    void InitElementSizes();
    void DoCacheLineTiling(int64_t alignedC, int32_t& startAxis, int64_t& clInnerProduct, int64_t& minUbFactor,
                           int64_t& outerProduct);
    void DoTiling(int64_t alignedC, int32_t startAxis, int64_t minUbFactor, int64_t clInnerProduct,
                  int64_t outerProduct);
    ge::graphStatus SetTilingData();

    gert::TilingContext* context_;
    uint32_t ubSize_{0};
    uint32_t coreNum_{0};
    uint32_t ubBlockSize_{0};
    uint32_t cacheLineSize_{0};

    int64_t inShape_[AXIS_COUNT]{0};
    int64_t outShape_[AXIS_COUNT]{0};
    int64_t blockSize_{0};
    int64_t crops_[CROPS_SIZE]{0};
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

ge::graphStatus BatchToSpaceTiling::GetSocInfo()
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

ge::graphStatus BatchToSpaceTiling::GetParams()
{
    // get data type size
    auto inputValueDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
    auto inputDataType = inputValueDesc->GetDataType();
    dSize_ = ge::GetSizeByDataType(inputDataType);
    OP_CHECK_IF(dSize_ <= 0, OP_LOGE(context_, "data size should be positive"), return ge::GRAPH_FAILED);

    // get input shape
    auto xInputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    auto xShape = xInputShape->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != AXIS_COUNT, OP_LOGE(context_, "x must be 4D"), return ge::GRAPH_FAILED);
    for (size_t i = 0; i < AXIS_COUNT; ++i) {
        inShape_[i] = xShape.GetDim(i);
    }

    // get block_size attribute
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockSizePtr);
    blockSize_ = *blockSizePtr;
    OP_CHECK_IF(blockSize_ <= 0, OP_LOGE(context_, "block_size must be positive"), return ge::GRAPH_FAILED);

    // get crops values (default to 0 if const data not available, e.g. TTK const mode)
    gert::Shape cropsShape;
    if (Ops::Base::GetConstIntToShape(context_, INPUT_IDX_CROPS, cropsShape)) {
        for (size_t i = 0; i < CROPS_SIZE; ++i) {
            crops_[i] = cropsShape[i];
            OP_CHECK_IF(crops_[i] < 0, OP_LOGE(context_, "crops must be non-negative"), return ge::GRAPH_FAILED);
        }
    } else {
        for (size_t i = 0; i < CROPS_SIZE; ++i) {
            crops_[i] = 0;
        }
    }

    // derive outShape
    int64_t bs2 = blockSize_ * blockSize_;
    OP_CHECK_IF(inShape_[AXIS_N] % bs2 != 0, OP_LOGE(context_, "N must be divisible by block_size^2"),
                return ge::GRAPH_FAILED);
    outShape_[AXIS_N] = inShape_[AXIS_N] / bs2;
    outShape_[AXIS_H] = inShape_[AXIS_H] * blockSize_ - crops_[0] - crops_[1];
    outShape_[AXIS_W] = inShape_[AXIS_W] * blockSize_ - crops_[2] - crops_[3];
    outShape_[AXIS_C] = inShape_[AXIS_C];

    OP_CHECK_IF(outShape_[AXIS_N] <= 0, OP_LOGE(context_, "N_out must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShape_[AXIS_H] <= 0, OP_LOGE(context_, "H_out must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShape_[AXIS_W] <= 0, OP_LOGE(context_, "W_out must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShape_[AXIS_C] <= 0, OP_LOGE(context_, "C must be positive"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void BatchToSpaceTiling::InitElementSizes()
{
    bufferSize_ = std::min(ubSize_ / BUFFER_NUM, MAX_BUFFER_SIZE);
    bufferSizeElements_ = bufferSize_ / static_cast<uint32_t>(dSize_);
    ubBlockElements_ = ubBlockSize_ / static_cast<uint32_t>(dSize_);
    cacheLineElements_ = cacheLineSize_ / static_cast<uint32_t>(dSize_);
}

void BatchToSpaceTiling::DoCacheLineTiling(int64_t alignedC, int32_t& startAxis, int64_t& clInnerProduct,
                                           int64_t& minUbFactor, int64_t& outerProduct)
{
    // find startAxis: first axis (from tail) where contiguous data ≥ CacheLine
    // clInnerProduct = product of axes AFTER startAxis (rolled up during search)
    startAxis = static_cast<int32_t>(AXIS_COUNT) - 1;
    clInnerProduct = 1;
    for (int32_t ax = static_cast<int32_t>(AXIS_COUNT) - 1; ax >= 0; --ax) {
        int64_t dimSize = (ax == static_cast<int32_t>(AXIS_C)) ? alignedC : outShape_[ax];
        if (dimSize * clInnerProduct >= static_cast<int64_t>(cacheLineElements_)) {
            startAxis = ax;
            break;
        }
        clInnerProduct *= dimSize;
    }

    // minUbFactor: minimum elements on startAxis to fill one CacheLine
    int64_t startAxisStep = (startAxis == static_cast<int32_t>(AXIS_C)) ? static_cast<int64_t>(ubBlockElements_) : 1;
    int64_t clMinFactor = Ops::Base::CeilDiv(static_cast<int64_t>(cacheLineElements_), clInnerProduct);
    minUbFactor = Ops::Base::CeilAlign(std::max(startAxisStep, clMinFactor), startAxisStep);

    // outerProduct: product of all axes before startAxis
    outerProduct = 1;
    for (int32_t ax = 0; ax < startAxis; ++ax) {
        outerProduct *= outShape_[ax];
    }
}

void BatchToSpaceTiling::DoTiling(int64_t alignedC, int32_t startAxis, int64_t minUbFactor, int64_t clInnerProduct,
                                  int64_t outerProduct)
{
    uint8_t bestAxis = AXIS_N;
    uint32_t bestFactor = 1;
    uint64_t bestTotalCount = 1;
    uint64_t bestRealCore = 1;

    int64_t cumInnerProduct = clInnerProduct;
    for (int32_t ax = startAxis; ax >= 0; --ax) {
        int64_t dimSize = (ax == static_cast<int32_t>(AXIS_C)) ? alignedC : outShape_[ax];
        int64_t step = (ax == static_cast<int32_t>(AXIS_C)) ? static_cast<int64_t>(ubBlockElements_) : 1;
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

ge::graphStatus BatchToSpaceTiling::SetTilingData()
{
    perCoreCount_ = Ops::Base::CeilDiv(totalCount_, static_cast<uint64_t>(coreNum_));
    uint64_t realCoreNum = Ops::Base::CeilDiv(totalCount_, perCoreCount_);

    OP_LOGI(context_, "BatchToSpace tiling: ubAxis=%u ubFactor=%u totalCount=%lu perCoreCount=%lu realCoreNum=%lu",
            ubAxis_, ubFactor_, totalCount_, perCoreCount_, realCoreNum);

    auto tilingData = context_->GetTilingData<BatchToSpaceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    for (size_t i = 0; i < AXIS_COUNT; ++i) {
        tilingData->inShape[i] = inShape_[i];
        tilingData->outShape[i] = outShape_[i];
    }
    tilingData->totalCount = totalCount_;
    tilingData->perCoreCount = perCoreCount_;
    tilingData->ubFactor = ubFactor_;
    tilingData->bufferSize = bufferSize_;
    tilingData->blockSize = blockSize_;
    tilingData->cropTop = crops_[0];
    tilingData->cropBottom = crops_[1];
    tilingData->cropLeft = crops_[2];
    tilingData->cropRight = crops_[3];

    const uint64_t tilingKey = GET_TPL_TILING_KEY(ubAxis_);
    OP_LOGI(context_, "tilingKey is %lu, ubAxis %u", tilingKey, ubAxis_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceTiling::ComputeAndSetTiling()
{
    InitElementSizes();

    int64_t alignedC = Ops::Base::CeilAlign(outShape_[AXIS_C], static_cast<int64_t>(ubBlockElements_));
    int32_t startAxis;
    int64_t clInnerProduct, minUbFactor, outerProduct;
    DoCacheLineTiling(alignedC, startAxis, clInnerProduct, minUbFactor, outerProduct);

    DoTiling(alignedC, startAxis, minUbFactor, clInnerProduct, outerProduct);
    return SetTilingData();
}

ge::graphStatus BatchToSpaceTiling::DoTiling()
{
    auto ret = GetSocInfo();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetSocInfo failed"), return ret);
    ret = GetParams();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetParams failed"), return ret);
    ret = ComputeAndSetTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ComputeAndSetTiling failed"), return ret);

    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BatchToSpaceTilingFunc(gert::TilingContext* context)
{
    BatchToSpaceTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingParseForBatchToSpace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BatchToSpace)
    .Tiling(BatchToSpaceTilingFunc)
    .TilingInputsDataDependency({INPUT_IDX_CROPS})
    .TilingParse<BatchToSpaceCompileInfo>(TilingParseForBatchToSpace);
} // namespace optiling
