/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "register/op_impl_registry.h"
#include "conversion/space_to_batch/op_kernel/arch35/space_to_batch_tiling_data.h"
#include "conversion/space_to_batch/op_kernel/arch35/space_to_batch_tiling_key.h"
#include "platform/platform_ascendc.h"
#include "op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"

namespace optiling {

static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_PADDINGS = 1;
static constexpr int64_t ATTR_IDX_BLOCK_SIZE = 0;
static constexpr size_t PADDINGS_ROWS = 2;
static constexpr size_t PADDINGS_COLS = 2;
static constexpr size_t PADDINGS_ELEM_COUNT = PADDINGS_ROWS * PADDINGS_COLS;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t MAX_BUFFER_SIZE = 64 * 1024U;

class SpaceToBatchTiling {
public:
    explicit SpaceToBatchTiling(gert::TilingContext* context) : context_(context){};
    ~SpaceToBatchTiling() = default;

    ge::graphStatus DoTiling();

private:
    ge::graphStatus ParamCheck();
    ge::graphStatus GetSocInfo();
    ge::graphStatus ComputeOutShape();
    ge::graphStatus DoOpTiling();

    int32_t dSize_{0};

    // input shapes
    int64_t inShape_[STB_AXIS_COUNT]{};
    int64_t outShape_[STB_AXIS_COUNT]{};
    int64_t blockSize_{0};
    int64_t paddings_[PADDINGS_ROWS][PADDINGS_COLS]{};

    // soc info
    uint32_t ubSize_{0};
    uint32_t ubBlockSize_{0};
    uint32_t coreNum_{0};
    uint32_t cacheLineSize_{0};
    uint32_t ubBlockElements_{0};
    uint32_t cacheLineElements_{0};

    // tiling result
    uint32_t realCoreNum_{0};
    uint8_t ubAxis_{0};
    uint32_t ubFactor_{0};
    uint64_t totalCount_{0};
    uint32_t bufferSize_{0};

    gert::TilingContext* context_;
};

template <typename T>
static std::string ArrToStr(const T* v, size_t size)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0)
            oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

ge::graphStatus SpaceToBatchTiling::ParamCheck()
{
    // x
    auto inputDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dSize_ = ge::GetSizeByDataType(inputDesc->GetDataType());
    OP_CHECK_IF(dSize_ <= 0, OP_LOGE(context_, "data size should be positive"), return ge::GRAPH_FAILED);

    auto xShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    OP_CHECK_IF(xStorageShape.GetDimNum() != STB_AXIS_COUNT,
                OP_LOGE(context_, "x must be 4D, got %zu", xStorageShape.GetDimNum()), return ge::GRAPH_FAILED);
    for (size_t i = 0; i < STB_AXIS_COUNT; ++i) {
        inShape_[i] = xStorageShape.GetDim(i);
    }

    // block_size attribute
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto blockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockSizePtr);
    blockSize_ = *blockSizePtr;
    OP_CHECK_IF(blockSize_ <= 0, OP_LOGE(context_, "block_size must be positive, got %ld", blockSize_),
                return ge::GRAPH_FAILED);

    // paddings
    gert::Shape paddings;
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context_, INPUT_IDX_PADDINGS, paddings),
                OP_LOGE(context_, "get paddings tensor failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(paddings.GetDimNum() != PADDINGS_ELEM_COUNT,
                OP_LOGE(context_, "paddings must have 4 elements, got %zu", paddings.GetDimNum()),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < PADDINGS_ROWS; ++i) {
        for (size_t j = 0; j < PADDINGS_COLS; ++j) {
            paddings_[i][j] = paddings[i * PADDINGS_COLS + j];
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchTiling::ComputeOutShape()
{
    int64_t padTop = paddings_[0][0];
    int64_t padBottom = paddings_[0][1];
    int64_t padLeft = paddings_[1][0];
    int64_t padRight = paddings_[1][1];

    int64_t hPadded = inShape_[STB_AXIS_H] + padTop + padBottom;
    int64_t wPadded = inShape_[STB_AXIS_W] + padLeft + padRight;

    OP_CHECK_IF(hPadded % blockSize_ != 0,
                OP_LOGE(context_, "H_padded (%ld) not divisible by block_size (%ld)", hPadded, blockSize_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(wPadded % blockSize_ != 0,
                OP_LOGE(context_, "W_padded (%ld) not divisible by block_size (%ld)", wPadded, blockSize_),
                return ge::GRAPH_FAILED);

    outShape_[STB_AXIS_N] = inShape_[STB_AXIS_N] * blockSize_ * blockSize_;
    outShape_[STB_AXIS_H] = hPadded / blockSize_;
    outShape_[STB_AXIS_W] = wPadded / blockSize_;
    outShape_[STB_AXIS_C] = inShape_[STB_AXIS_C];

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchTiling::GetSocInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    realCoreNum_ = coreNum_;
    OP_CHECK_IF(coreNum_ == 0U, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0U, OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    ubSize_ = static_cast<uint32_t>(ubSize);

    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);

    cacheLineElements_ = cacheLineSize_ / static_cast<uint32_t>(dSize_);
    ubBlockElements_ = ubBlockSize_ / static_cast<uint32_t>(dSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchTiling::DoOpTiling()
{
    bufferSize_ = ubSize_ / BUFFER_NUM;
    bufferSize_ = std::min(bufferSize_, MAX_BUFFER_SIZE);
    uint32_t maxUBElements = bufferSize_ / static_cast<uint32_t>(dSize_);

    // Align tail axis to ubBlock
    int64_t alignedC = Ops::Base::CeilAlign(outShape_[STB_AXIS_C], static_cast<int64_t>(ubBlockElements_));

    // Step 1: find startAxis (first axis from C where dimSize * clInnerProduct >= cacheLineElements)
    int32_t startAxis = STB_AXIS_C;
    int64_t clInnerProduct = 1;
    for (int32_t ax = STB_AXIS_C; ax >= 0; --ax) {
        int64_t dimSize = (ax == STB_AXIS_C) ? alignedC : outShape_[ax];
        if (dimSize * clInnerProduct >= static_cast<int64_t>(cacheLineElements_)) {
            startAxis = ax;
            break;
        }
        clInnerProduct *= dimSize;
    }

    int64_t step = (startAxis == STB_AXIS_C) ? static_cast<int64_t>(ubBlockElements_) : 1;
    int64_t minUbFactor = Ops::Base::CeilAlign(
        std::max(step, Ops::Base::CeilDiv(static_cast<int64_t>(cacheLineElements_), clInnerProduct)), step);

    // outerProduct = product of outShape[0 .. startAxis-1]
    int64_t outerProduct = 1;
    for (int32_t i = 0; i < startAxis; ++i) {
        outerProduct *= outShape_[i];
    }

    // Step 2: traverse from startAxis outward
    int64_t cumInnerProduct = clInnerProduct;
    int32_t bestAxis = STB_AXIS_N;
    int64_t bestFactor = 1;
    uint64_t bestTotalCount = 1;
    uint32_t bestRealCore = 1;

    bool found = false;
    for (int32_t ax = startAxis; ax >= 0; --ax) {
        int64_t dimSize = (ax == STB_AXIS_C) ? alignedC : outShape_[ax];
        int64_t factorStep = (ax == STB_AXIS_C) ? static_cast<int64_t>(ubBlockElements_) : 1;
        int64_t factorStart = (ax == startAxis) ? minUbFactor : factorStep;

        for (int64_t factor = factorStart; factor <= dimSize; factor += factorStep) {
            int64_t blockSize = factor * cumInnerProduct;
            if (blockSize > static_cast<int64_t>(maxUBElements)) {
                break;
            }

            uint64_t totalCount = static_cast<uint64_t>(outerProduct * Ops::Base::CeilDiv(dimSize, factor));
            uint64_t perCoreCount = Ops::Base::CeilDiv(totalCount, static_cast<uint64_t>(coreNum_));
            uint32_t realCoreNum = static_cast<uint32_t>(Ops::Base::CeilDiv(totalCount, perCoreCount));

            double coreRatio = static_cast<double>(realCoreNum) / static_cast<double>(coreNum_);
            if (coreRatio >= 0.8 || realCoreNum > bestRealCore) {
                bestAxis = ax;
                bestFactor = factor;
                bestTotalCount = totalCount;
                found = true;
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

    if (!found) {
        // Fallback: cut N axis, factor=1, totalCount=1
        bestAxis = STB_AXIS_N;
        bestFactor = 1;
        bestTotalCount = 1;
    }

    ubAxis_ = static_cast<uint8_t>(bestAxis);
    ubFactor_ = static_cast<uint32_t>(bestFactor);
    totalCount_ = bestTotalCount;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchTiling::DoTiling()
{
    auto ret = ParamCheck();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParamCheck failed"), return ret);

    ret = ComputeOutShape();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ComputeOutShape failed"), return ret);

    ret = GetSocInfo();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetSocInfo failed"), return ret);

    ret = DoOpTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "DoOpTiling failed"), return ret);

    // Fill TilingData
    auto tilingData = context_->GetTilingData<SpaceToBatchTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    for (size_t i = 0; i < STB_AXIS_COUNT; ++i) {
        tilingData->inShape[i] = inShape_[i];
        tilingData->outShape[i] = outShape_[i];
    }
    tilingData->blockSize = blockSize_;
    for (size_t i = 0; i < PADDINGS_ROWS; ++i) {
        for (size_t j = 0; j < PADDINGS_COLS; ++j) {
            tilingData->paddings[i][j] = paddings_[i][j];
        }
    }
    tilingData->totalCount = totalCount_;
    tilingData->bufferSize = bufferSize_;
    tilingData->ubAxis = ubAxis_;
    tilingData->ubFactor = ubFactor_;

    uint64_t perCoreCount = Ops::Base::CeilDiv(totalCount_, static_cast<uint64_t>(coreNum_));
    tilingData->perCoreCount = perCoreCount;
    realCoreNum_ = static_cast<uint32_t>(Ops::Base::CeilDiv(totalCount_, perCoreCount));

    const uint64_t tilingKey = GET_TPL_TILING_KEY(ubAxis_);
    OP_LOGI(context_, "tilingKey=%lu, ubAxis=%u, ubFactor=%u, totalCount=%lu, perCoreCount=%lu, bufferSize=%u",
            tilingKey, ubAxis_, ubFactor_, totalCount_, perCoreCount, bufferSize_);
    OP_LOGI(context_, "inShape=%s, outShape=%s, blockSize=%ld, paddings=[%ld,%ld,%ld,%ld]",
            ArrToStr(inShape_, STB_AXIS_COUNT).c_str(), ArrToStr(outShape_, STB_AXIS_COUNT).c_str(), blockSize_,
            paddings_[0][0], paddings_[0][1], paddings_[1][0], paddings_[1][1]);

    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum_);

    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SpaceToBatch(gert::TilingContext* context)
{
    SpaceToBatchTiling tiling{context};
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForSpaceToBatch([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SpaceToBatch)
    .Tiling(Tiling4SpaceToBatch)
    .TilingInputsDataDependency({INPUT_IDX_PADDINGS})
    .TilingParse<SpaceToBatchCompileInfo>(TilingPrepareForSpaceToBatch);

} // namespace optiling
