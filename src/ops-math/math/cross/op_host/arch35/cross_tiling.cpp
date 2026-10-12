/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cross_tiling.h"

using namespace ge;

namespace optiling {
static constexpr uint64_t INPUT_X1 = 0;
static constexpr uint64_t INPUT_X2 = 1;
static constexpr uint64_t DIM = 0;
static constexpr int64_t INT_MAX = 2147483647;

ge::graphStatus CrossTiling::GetPlatformInfo()
{
    OP_LOGD(context_, "CrossTiling GetPlatformInfo.");
    compileInfo_ = static_cast<const CrossCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossTiling::CheckBaseShapeAndAttrs()
{
    OP_LOGD(context_, "CrossTiling CheckBaseShapeAndAttrs.");
    const gert::StorageShape* shape1 = context_->GetInputShape(INPUT_X1);
    const gert::StorageShape* shape2 = context_->GetInputShape(INPUT_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shape1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shape2);

    auto s1 = shape1->GetStorageShape();
    auto s2 = shape2->GetStorageShape();
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF((attrs == nullptr), OP_LOGE(context_, "Get attrs Failed."), return ge::GRAPH_FAILED);

    int64_t dim = *(attrs->GetAttrPointer<int64_t>(DIM));
    dimNum1_ = s1.GetDimNum();
    dimNum2_ = s2.GetDimNum();

    for (int64_t i = 0; i < dimNum1_; i++) {
        x1Dims_[i] = s1.GetDim(i);
    }
    for (int64_t i = 0; i < dimNum2_; i++) {
        x2Dims_[i] = s2.GetDim(i);
    }

    OP_CHECK_IF((dim < -dimNum1_ || dim >= dimNum1_),
                OP_LOGE(context_, "dim must be in [%ld, %ld], dim: [%ld].", -dimNum1_, dimNum1_ - 1, dim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((dimNum1_ != dimNum2_),
                OP_LOGE(context_, "x1 and x2 dim count mismatch: %ld vs %ld.", dimNum1_, dimNum2_),
                return ge::GRAPH_FAILED);

    int64_t tempDimNum = dimNum1_ == 0 ? 1 : dimNum1_;
    normalizedDim_ = (dim + tempDimNum) % tempDimNum;

    int64_t dimSize1 = x1Dims_[normalizedDim_];
    int64_t dimSize2 = x2Dims_[normalizedDim_];
    OP_CHECK_IF((dimSize1 != 3), OP_LOGE(context_, "x1 dim[%ld] must be 3, got %ld.", normalizedDim_, dimSize1),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((dimSize2 != 3), OP_LOGE(context_, "x2 dim[%ld] must be 3, got %ld.", normalizedDim_, dimSize2),
                return ge::GRAPH_FAILED);

    {
        int64_t origDimNum = dimNum1_;
        int64_t newDimNum = 0;
        int64_t curProd = 1;
        int64_t newNormalizedDim = 0;
        for (int64_t i = 0; i < origDimNum; i++) {
            bool needNewAxis = (x1Dims_[i] != x2Dims_[i]) || (i == normalizedDim_);
            if (needNewAxis) {
                if (curProd > 1) {
                    x1Dims_[newDimNum] = curProd;
                    x2Dims_[newDimNum] = curProd;
                    newDimNum++;
                }
                x1Dims_[newDimNum] = x1Dims_[i];
                x2Dims_[newDimNum] = x2Dims_[i];
                if (i == normalizedDim_)
                    newNormalizedDim = newDimNum;
                newDimNum++;
                curProd = 1;
            } else {
                curProd *= x1Dims_[i];
            }
        }
        if (curProd > 1) {
            x1Dims_[newDimNum] = curProd;
            x2Dims_[newDimNum] = curProd;
            newDimNum++;
        }
        for (int64_t i = newDimNum; i < MAX_DIM; i++) {
            x1Dims_[i] = 1;
            x2Dims_[i] = 1;
        }
        dimNum1_ = newDimNum;
        dimNum2_ = newDimNum;
        normalizedDim_ = newNormalizedDim;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossTiling::CheckBroadcastAndMergeShape()
{
    OP_LOGD(context_, "CrossTiling CheckBroadcastAndMergeShape.");
    for (int64_t i = 0; i < dimNum1_; i++) {
        if (i == normalizedDim_)
            continue;
        int64_t size1 = x1Dims_[i];
        int64_t size2 = x2Dims_[i];
        OP_CHECK_IF((size1 != size2 && size1 != 1 && size2 != 1),
                    OP_LOGE(context_, "Shapes not broadcastable at dim %ld: %ld vs %ld.", i, size1, size2),
                    return ge::GRAPH_FAILED);
    }

    dim_ = normalizedDim_;
    dimNum_ = dimNum1_;
    ySize_ = 1;
    for (int64_t i = 0; i < dimNum1_; i++) {
        if (i == normalizedDim_) {
            mergedShape_[i] = 1;
            ySize_ *= x1Dims_[i];
        } else {
            mergedShape_[i] = std::max(x1Dims_[i], x2Dims_[i]);
            ySize_ *= mergedShape_[i];
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossTiling::CalcStrideAndVectors()
{
    OP_LOGD(context_, "CrossTiling CalcStrideAndVectors.");
    int64_t stride[4] = {1, 1, 1, 1};
    for (int64_t i = dimNum1_ - 1; i >= 0; i--) {
        x1Stride_[i] = (x1Dims_[i] == 1) ? 0 : stride[0];
        x2Stride_[i] = (x2Dims_[i] == 1) ? 0 : stride[1];
        mergedStride_[i] = stride[2];
        yStride_[i] = stride[3];

        stride[0] *= x1Dims_[i];
        stride[1] *= x2Dims_[i];
        stride[2] *= mergedShape_[i];
        stride[3] *= (i == dim_ ? 3 : mergedShape_[i]);
    }

    dimStride_ = 1;
    for (int64_t i = normalizedDim_ + 1; i < dimNum1_; i++) {
        dimStride_ *= mergedShape_[i];
    }

    totalVectors_ = 1;
    for (int64_t i = 0; i < dimNum1_; i++) {
        if (i != normalizedDim_)
            totalVectors_ *= mergedShape_[i];
    }

    int64_t activeCount = 0;
    int64_t activeIdx[MAX_DIM];
    for (int64_t i = 0; i < dimNum_; i++) {
        if (i == dim_)
            continue;
        if (mergedShape_[i] > 1) {
            activeIdx[activeCount++] = i;
        }
    }
    tilingData_.activeDimCount = activeCount;
    for (int64_t i = 0; i < activeCount; i++) {
        tilingData_.activeDimIndices[i] = activeIdx[i];
    }
    for (int64_t i = activeCount; i < MAX_DIM; i++) {
        tilingData_.activeDimIndices[i] = 0;
    }

    tilingData_.usedInt64 = ySize_ > INT_MAX;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossTiling::GetShapeAttrsInfo()
{
    if (CheckBaseShapeAndAttrs() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CheckBroadcastAndMergeShape() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CalcStrideAndVectors() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossTiling::DoOpTiling()
{
    OP_LOGD(context_, "CrossTiling DoOpTiling.");

    int64_t coreNum = compileInfo_->coreNum;

    // Multi-block-per-core ladder: pick the largest blocksPerCore whose resulting
    // per-block workload still clears MIN_VECTORS_PER_BLOCK. Formulated in terms
    // of "average work per core" (totalVectors_ / coreNum) so the threshold reads
    // the same way the kernel-side comment does ("each block ≥ ~250K vectors").
    int64_t avgVectorsPerCore = (coreNum > 0) ? totalVectors_ / coreNum : totalVectors_;
    int64_t blocksPerCore = 1;
    if (avgVectorsPerCore >= CrossConst::MAX_BLOCKS_PER_CORE * CrossConst::MIN_VECTORS_PER_BLOCK) {
        blocksPerCore = CrossConst::MAX_BLOCKS_PER_CORE;
    } else if (avgVectorsPerCore >= CrossConst::MEDIUM_BLOCKS_PER_CORE * CrossConst::MIN_VECTORS_PER_BLOCK) {
        blocksPerCore = CrossConst::MEDIUM_BLOCKS_PER_CORE;
    }
    int64_t totalBlocks = coreNum * blocksPerCore;
    if (totalBlocks > totalVectors_)
        totalBlocks = totalVectors_;

    int64_t vectorsPerBlock = totalVectors_ / totalBlocks;
    int64_t formerBlock = totalVectors_ % totalBlocks;

    tilingData_.totalVectors = totalVectors_;
    tilingData_.coreNum = coreNum;
    tilingData_.dim = dim_;
    tilingData_.dimNum = dimNum_;
    tilingData_.dimStride = dimStride_;
    tilingData_.blocksPerCore = blocksPerCore;
    tilingData_.formerBlock = formerBlock;
    tilingData_.vectorsPerBlock = vectorsPerBlock;

    for (int64_t i = 0; i < MAX_DIM; i++) {
        if (i < dimNum_) {
            tilingData_.mergedStride[i] = mergedStride_[i];
            tilingData_.x1Stride[i] = x1Stride_[i];
            tilingData_.x2Stride[i] = x2Stride_[i];
            tilingData_.yStride[i] = yStride_[i];
        } else {
            tilingData_.mergedStride[i] = 1;
            tilingData_.x1Stride[i] = 0;
            tilingData_.x2Stride[i] = 0;
            tilingData_.yStride[i] = 0;
        }
    }

    blockDim_ = totalBlocks;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CrossTiling::PostTiling()
{
    OP_LOGD(context_, "CrossTiling PostTiling.");

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0;

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           &tilingData_, sizeof(CrossRegbaseTilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(CrossRegbaseTilingData));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Cross(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4Cross start.");

    CrossTiling crossTiling(context);
    auto ret = crossTiling.DoTiling();
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGD(context, "Tiling4Cross failed!"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "Tiling4Cross end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4CrossAscendc(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4CrossAscendc.");

    auto compileInfo = context->GetCompiledInfo<CrossCompileInfo>();

    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "core num is negative."),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4CrossAscendc.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Cross(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<CrossCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD("TilingPrepare4Cross", "Ascend C TilingPrepare4Cross success.");
    return TilingPrepare4CrossAscendc(context);
}

IMPL_OP_OPTILING(Cross).Tiling(Tiling4Cross).TilingParse<CrossCompileInfo>(TilingPrepare4Cross);
} // namespace optiling
