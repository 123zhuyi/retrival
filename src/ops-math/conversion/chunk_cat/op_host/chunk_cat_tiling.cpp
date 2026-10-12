/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_cat_tiling.cpp
 * \brief
 */

#include "chunk_cat_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base_util.h"

namespace optiling {

constexpr uint32_t UB_BLOCK_SIZE = 32;    // UB块大小
constexpr uint32_t TRANS_BLOCK = 16;      // 转置行数
constexpr uint32_t RESERVE_UB = 256;      // 接口获取UB的预留空间
constexpr uint32_t HALF = 2;              // 半对齐/UB对半切分
constexpr uint32_t ONETHIRD = 3;          // UB对三切分
constexpr uint32_t DEFAUL_TILING_KEY = 0; // 默认tiling key
constexpr uint32_t NUM_THIRTY_TWO = 32;

static const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

std::string ChunkCatTiling::TilingDataToString() const
{
    return "blockRowNum = " + std::to_string(blockRowNum_) + ", blockColNum = " + std::to_string(blockColNum_) +
           ", dim = " + std::to_string(dim_) + ", numChunk = " + std::to_string(numChunk_) +
           ", outputRow = " + std::to_string(outputRow_) + ", outputCol = " + std::to_string(outputCol_) +
           ", blockRowFactor = " + std::to_string(blockRowFactor_) +
           ", blockColFactor = " + std::to_string(blockColFactor_) +
           ", tailBlockRowFactor = " + std::to_string(tailBlockRowFactor_) +
           ", tailBlockColFactor = " + std::to_string(tailBlockColFactor_) +
           ", ubRowFactor = " + std::to_string(ubRowFactor_) + ", ubColFactor = " + std::to_string(ubColFactor_) +
           ", inputNum = " + std::to_string(inputNum_) + ", inUbSize = " + std::to_string(inUbSize_) +
           ", outUbSize = " + std::to_string(outUbSize_) + ", isAllAlign = " + std::to_string(isAllAlign_) +
           ", isHalfAlign = " + std::to_string(isHalfAlign_) + ", isOneConcat = " + std::to_string(isOneConcat_);
}

// 获取硬件信息
ge::graphStatus ChunkCatTiling::GetPlatformInfo()
{
    // 获取ubsize coreNum_
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0, OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    isRegBase = Ops::Base::IsRegbaseSocVersion(context_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkCatTiling::GetInputInfo()
{
    // 获取属性
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    const int64_t* dimPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimPtr);
    dim_ = static_cast<int64_t>(*dimPtr);
    OP_CHECK_IF(dim_ != 0, OP_LOGE(context_, "dim must be 0 now"), return ge::GRAPH_FAILED);
    const int64_t* numChunkPtr = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numChunkPtr);
    numChunk_ = static_cast<int64_t>(*numChunkPtr);
    OP_CHECK_IF(numChunk_ <= 0, OP_LOGE(context_, "numChunk must be greater than 0 now"), return ge::GRAPH_FAILED);

    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, anchorInstanceInfo);
    inputNum_ = anchorInstanceInfo->GetInstanceNum();
    OP_CHECK_IF(inputNum_ == 0, OP_LOGE(context_, "input num can not be 0"), return ge::GRAPH_FAILED);
    // 获取数据类型
    auto inputDesc = context_->GetDynamicInputDesc(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto inputDataType = inputDesc->GetDataType();
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputDataType = outputDesc->GetDataType();
    OP_CHECK_IF(supportedDtype.count(inputDataType) == 0, OP_LOGE(context_, "input dtype is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(supportedDtype.count(outputDataType) == 0, OP_LOGE(context_, "output dtype is invalid"),
                return ge::GRAPH_FAILED);
    if (inputDataType == ge::DT_FLOAT && outputDataType != ge::DT_FLOAT) {
        OP_LOGE(context_, "output dtype must be float when input dtype is float");
        return ge::GRAPH_FAILED;
    }
    srcDtypeSize_ = ge::GetSizeByDataType(inputDataType);
    OP_CHECK_IF(srcDtypeSize_ == 0, OP_LOGE(context_, "input dtype size can not be 0"), return ge::GRAPH_FAILED);
    srcEleUbBlock_ = UB_BLOCK_SIZE / srcDtypeSize_;
    uint32_t reserveUb = isRegBase ? 0 : RESERVE_UB;
    if (inputDataType != outputDataType) {
        inUbSize_ = (ubSize_ + reserveUb) / ONETHIRD;
    } else {
        inUbSize_ = (ubSize_ + reserveUb) / HALF;
    }
    outUbSize_ = (ubSize_ + reserveUb) - inUbSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkCatTiling::CalculateOutputInfo()
{
    for (int64_t i = 0; i < inputNum_; i++) {
        auto inputTensorShapePtr = context_->GetDynamicInputShape(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputTensorShapePtr);
        gert::Shape inputTensorShape = inputTensorShapePtr->GetStorageShape();

        int64_t chunkDimSize = inputTensorShape.GetDim(dim_);
        int64_t chunkCol = (chunkDimSize + numChunk_ - 1) / numChunk_;

        int64_t dim0Size = numChunk_;
        int64_t dim1Size = chunkCol;
        int64_t inputTensorDimNum = inputTensorShape.GetDimNum();
        for (int64_t j = 0; j < dim_; j++) {
            dim0Size *= inputTensorShape.GetDim(j);
        }
        for (int64_t j = dim_ + 1; j < inputTensorDimNum; j++) {
            dim1Size *= inputTensorShape.GetDim(j);
        }
        if (i == 0) {
            outputRow_ = dim0Size;
        } else if (outputRow_ != dim0Size) {
            OP_LOGE(context_, "op expects same sizes of 0,...,dim-1 dimensions for all tensors");
            return ge::GRAPH_FAILED;
        }
        if (dim1Size % srcEleUbBlock_ != 0) {
            isAllAlign_ = false;
        }
        if (dim1Size % (srcEleUbBlock_ / 2) != 0) {
            isHalfAlign_ = false;
        }
        outputCol_ += dim1Size;
    }
    isOneConcat_ = (outputRow_ == 1) && (!isAllAlign_);
    return ge::GRAPH_SUCCESS;
}

void ChunkCatTiling::DoUbSplit()
{
    if (isRegBase) {
        // 列切
        uint32_t colLimit = inUbSize_ / srcDtypeSize_;
        colLimit = colLimit - NUM_THIRTY_TWO * srcEleUbBlock_;
        int64_t ubColLoop = (outputCol_ + colLimit - 1) / colLimit;
        ubColFactor_ = (outputCol_ + ubColLoop - 1) / ubColLoop;
        ubColFactor_ = (ubColFactor_ + srcEleUbBlock_ - 1) / srcEleUbBlock_ * srcEleUbBlock_;
        ubColFactor_ = ubColFactor_ > colLimit ? colLimit : ubColFactor_;
        // 行切
        uint32_t rowLimit = ubColLoop == 1 ? colLimit / ubColFactor_ : 1;
        int64_t ubRowLoop = (outputRow_ + rowLimit - 1) / rowLimit;
        ubRowFactor_ = (outputRow_ + ubRowLoop - 1) / ubRowLoop;
    } else if (isAllAlign_ || isOneConcat_) {
        // 列切
        uint32_t colLimit = inUbSize_ / srcDtypeSize_;
        colLimit = (isOneConcat_) ? colLimit - 32 * (srcEleUbBlock_ - 1) : colLimit;
        int64_t ubColLoop = (outputCol_ + colLimit - 1) / colLimit;
        ubColFactor_ = (outputCol_ + ubColLoop - 1) / ubColLoop;
        ubColFactor_ = (ubColFactor_ + srcEleUbBlock_ - 1) / srcEleUbBlock_ * srcEleUbBlock_;
        ubColFactor_ = ubColFactor_ > colLimit ? colLimit : ubColFactor_;
        // 行切
        uint32_t rowLimit = ubColLoop == 1 ? colLimit / ubColFactor_ : 1;
        int64_t ubRowLoop = (outputRow_ + rowLimit - 1) / rowLimit;
        ubRowFactor_ = (outputRow_ + ubRowLoop - 1) / ubRowLoop;
    } else {
        // 行切
        uint32_t rowLimit = isHalfAlign_ ? TRANS_BLOCK * HALF : TRANS_BLOCK * srcEleUbBlock_;
        int64_t ubRowLoop = (outputRow_ + rowLimit - 1) / rowLimit;
        ubRowFactor_ = (outputRow_ + ubRowLoop - 1) / ubRowLoop;
        // 列切
        uint32_t colLimit = inUbSize_ / rowLimit / srcDtypeSize_ - 2 * srcEleUbBlock_;
        int64_t ubColLoop = (outputCol_ + colLimit - 1) / colLimit;
        ubColFactor_ = (outputCol_ + ubColLoop - 1) / ubColLoop;
        ubColFactor_ = (ubColFactor_ + srcEleUbBlock_ - 1) / srcEleUbBlock_ * srcEleUbBlock_;
        ubColFactor_ = ubColFactor_ > colLimit ? colLimit : ubColFactor_;
    }
}

void ChunkCatTiling::DoBlockSplit()
{
    // 行切
    blockRowFactor_ = (outputRow_ + coreNum_ - 1) / coreNum_;
    blockRowFactor_ = (blockRowFactor_ + ubRowFactor_ - 1) / ubRowFactor_ * ubRowFactor_;
    blockRowNum_ = (outputRow_ + blockRowFactor_ - 1) / blockRowFactor_;
    tailBlockRowFactor_ = outputRow_ - blockRowFactor_ * (blockRowNum_ - 1);
    tailBlockRowFactor_ = tailBlockRowFactor_ == 0 ? blockRowFactor_ : tailBlockRowFactor_;
    int64_t leftCoreNum = coreNum_ / blockRowNum_;
    // 列切
    blockColFactor_ = (outputCol_ + leftCoreNum - 1) / leftCoreNum;
    blockColFactor_ = (blockColFactor_ + ubColFactor_ - 1) / ubColFactor_ * ubColFactor_;
    blockColNum_ = (outputCol_ + blockColFactor_ - 1) / blockColFactor_;
    tailBlockColFactor_ = outputCol_ - blockColFactor_ * (blockColNum_ - 1);
    tailBlockColFactor_ = tailBlockColFactor_ == 0 ? blockColFactor_ : tailBlockColFactor_;
    usedCoreNum_ = blockRowNum_ * blockColNum_;
}

void ChunkCatTiling::SetTilingData(ChunkCatTilingData* tilingData)
{
    tilingData->isAllAlign = isAllAlign_;
    tilingData->isHalfAlign = isHalfAlign_;
    tilingData->isOneConcat = isOneConcat_;
    tilingData->dim = dim_;
    tilingData->numChunk = numChunk_;
    tilingData->outputRow = outputRow_;
    tilingData->outputCol = outputCol_;
    tilingData->blockRowFactor = blockRowFactor_;
    tilingData->blockColFactor = blockColFactor_;
    tilingData->tailBlockRowFactor = tailBlockRowFactor_;
    tilingData->tailBlockColFactor = tailBlockColFactor_;
    tilingData->ubRowFactor = ubRowFactor_;
    tilingData->ubColFactor = ubColFactor_;
    tilingData->inputNum = inputNum_;
    tilingData->inUbSize = inUbSize_;
    tilingData->outUbSize = outUbSize_;
    tilingData->blockRowNum = blockRowNum_;
    tilingData->blockColNum = blockColNum_;

    size_t* workspaceSize = context_->GetWorkspaceSizes(1);
    *workspaceSize = sysWorkspaceSize_;
    context_->SetTilingKey(DEFAUL_TILING_KEY);
    context_->SetBlockDim(usedCoreNum_);
}

static ge::graphStatus Tiling4ChunkCat(gert::TilingContext* context)
{
    OP_LOGD(context, "ChunkCatTiling");
    ChunkCatTiling tiling(context);
    OP_CHECK_IF(tiling.GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.GetInputInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetInputInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.CalculateOutputInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateOutputInfo error"),
                return ge::GRAPH_FAILED);
    tiling.DoUbSplit();
    tiling.DoBlockSplit();

    ChunkCatTilingData* tilingData = context->GetTilingData<ChunkCatTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(memset_s(tilingData, sizeof(ChunkCatTilingData), 0, sizeof(ChunkCatTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling.SetTilingData(tilingData);
    OP_LOGD(context, "tiling data: %s", tiling.TilingDataToString().c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ChunkCat([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkCat).Tiling(Tiling4ChunkCat).TilingParse<ChunkCatCompileInfo>(TilingPrepare4ChunkCat);
} // namespace optiling
