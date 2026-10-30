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
 * \file matrix_diag_tiling_arch35.cpp
 * \brief implemention of MatrixDiagAsc tiling
 */
#include "matrix_diag_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "util/platform_util.h"
#include <cstring>

using namespace ge;

namespace optiling {
namespace MatrixDiagAsc {
static constexpr int64_t SIMT_SIZE = 1024;
static constexpr int64_t BLOCK_SIZE = 32;
static constexpr int64_t HALF_VL_LEN = 128;
static constexpr int64_t VL_LEN = 256;
static constexpr int64_t USED_MIN_UB_SIZE = 8 * 1024; // 8KB
static constexpr int64_t BASE_512 = 512;
static constexpr int64_t BASE_4 = 4;
static constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);
static constexpr int64_t B8_MAX_SIZE = 65536;

std::string MatrixDiagTiling::PrintTilingData()
{
    std::string tdStr;
    if (tilingKey_ == ::MatrixDiagAsc::TILING_PURE_COPY) {
        tdStr += std::to_string(tilingDataPureCopy_.size) + ",";
        tdStr += std::to_string(tilingDataPureCopy_.ubSize) + ",";
        tdStr += std::to_string(tilingDataPureCopy_.realCoreNum) + ",";
        tdStr += std::to_string(tilingDataPureCopy_.mainBlockCount) + ",";
        tdStr += std::to_string(tilingDataPureCopy_.mainBlockFactor) + ",";
        tdStr += std::to_string(tilingDataPureCopy_.tailBlockFactor) + ",";
    } else if (tilingKey_ == ::MatrixDiagAsc::TILING_SIMT) {
        tdStr += std::to_string(tilingDataSimt_.batchSize) + ",";
        tdStr += std::to_string(tilingDataSimt_.nSize) + ",";
        tdStr += std::to_string(tilingDataSimt_.ubSize) + ",";
        tdStr += std::to_string(tilingDataSimt_.realCoreNum) + ",";
        tdStr += std::to_string(tilingDataSimt_.mainBlockCount) + ",";
        tdStr += std::to_string(tilingDataSimt_.mainBlockFactor) + ",";
        tdStr += std::to_string(tilingDataSimt_.tailBlockFactor) + ",";
    } else if (tilingKey_ == ::MatrixDiagAsc::TILING_SCATTER_LARGE) {
        tdStr += std::to_string(tilingDataLarge_.realCoreNum) + ",";
        tdStr += std::to_string(tilingDataLarge_.batchSize) + ",";
        tdStr += std::to_string(tilingDataLarge_.nSize) + ",";
        tdStr += std::to_string(tilingDataLarge_.nUbFactor) + ",";
        tdStr += std::to_string(tilingDataLarge_.nUbTailFactor) + ",";
        tdStr += std::to_string(tilingDataLarge_.nUbCount) + ",";
        tdStr += std::to_string(tilingDataLarge_.blockFactor) + ",";
        tdStr += std::to_string(tilingDataLarge_.blockTailFactor) + ",";
        tdStr += std::to_string(tilingDataLarge_.blockMainCount) + ",";
    } else if (
        tilingKey_ == ::MatrixDiagAsc::TILING_SCATTER_HIGH || tilingKey_ == ::MatrixDiagAsc::TILING_SCATTER_LOW) {
        tdStr += std::to_string(tilingDataScatter_.realCoreNum) + ",";
        tdStr += std::to_string(tilingDataScatter_.batchSize) + ",";
        tdStr += std::to_string(tilingDataScatter_.nSize) + ",";
        tdStr += std::to_string(tilingDataScatter_.batchUbFactor) + ",";
        tdStr += std::to_string(tilingDataScatter_.batchUbTailFactor) + ",";
        tdStr += std::to_string(tilingDataScatter_.batchUbCount) + ",";
        tdStr += std::to_string(tilingDataScatter_.blockFactor) + ",";
        tdStr += std::to_string(tilingDataScatter_.blockTailFactor) + ",";
        tdStr += std::to_string(tilingDataScatter_.blockMainCount) + ",";
    }
    return tdStr;
}

template <typename T>
ge::graphStatus MatrixDiagTiling::SetTilingStruct(T& tilingSturct)
{
    auto ptrTD = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrTD);
    auto capSize = ptrTD->GetCapacity();
    void* ptrData = ptrTD->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingSturct);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingSturct)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    ptrTD->SetDataSize(sizeof(tilingSturct));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixDiagTiling::SetTilingData()
{
    switch (tilingKey_) {
        case ::MatrixDiagAsc::TILING_PURE_COPY:
            OP_CHECK_IF(
                SetTilingStruct<::MatrixDiagAsc::MatrixDiagPureCopyTilingData>(tilingDataPureCopy_) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set pure copy tiling struct is failed!"), return ge::GRAPH_FAILED);
            break;
        case ::MatrixDiagAsc::TILING_SIMT:
            OP_CHECK_IF(
                SetTilingStruct<::MatrixDiagAsc::MatrixDiagSimtTilingData>(tilingDataSimt_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set simt tiling struct is failed!"), return ge::GRAPH_FAILED);
            break;
        case ::MatrixDiagAsc::TILING_SCATTER_LARGE:
            OP_CHECK_IF(
                SetTilingStruct<::MatrixDiagAsc::MatrixDiagScatterLargeTilingData>(tilingDataLarge_) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set scatterLarge tiling struct is failed!"), return ge::GRAPH_FAILED);
            break;
        case ::MatrixDiagAsc::TILING_SCATTER_HIGH:
        case ::MatrixDiagAsc::TILING_SCATTER_LOW:
            OP_CHECK_IF(
                SetTilingStruct<::MatrixDiagAsc::MatrixDiagScatterTilingData>(tilingDataScatter_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set scatterHigh tiling struct is failed!"), return ge::GRAPH_FAILED);
            break;
        default:
            return ge::GRAPH_SUCCESS;
            break;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixDiagTiling::WriteTilingData()
{
    OP_CHECK_IF(
        context_->SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Set tiling key is failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->SetBlockDim(static_cast<uint32_t>(compileInfo_->coreNum)) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Set used core size is failed!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "set tiling data failed!"),
        return ge::GRAPH_FAILED);
    size_t totalWorkspaceSize = SYS_WORKSPACE_SIZE;
    size_t* ptrWS = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrWS);
    ptrWS[0] = totalWorkspaceSize;

    OP_LOGI(context_->GetNodeName(), "The tiling data is: %s", PrintTilingData().c_str());

    return ge::GRAPH_SUCCESS;
}

void MatrixDiagTiling::CalcSimtTilingData()
{
    int64_t sideLengthFactor = HALF_VL_LEN / xDtypeSize_;
    int64_t batchSize = fusedShape_[0];
    int64_t nSize = fusedShape_[1];
    int64_t coreNum = compileInfo_->coreNum;
    int64_t blockNum = Ops::Base::CeilDiv(batchSize * nSize, sideLengthFactor);
    int64_t cores = std::min(coreNum, blockNum);
    SetBlockSplitInfo(cores, 1, batchSize * nSize, 1); // SIMT只关注element，所以分核简化为一维，和纯搬运一致
    tilingDataSimt_.batchSize = batchSize;
    tilingDataSimt_.ubSize = ubSize_;
    tilingDataSimt_.nSize = nSize;
    tilingDataSimt_.realCoreNum = cores;
    tilingDataSimt_.mainBlockCount = mBlockFactorCount_;
    tilingDataSimt_.mainBlockFactor = mBlockFactor_;
    tilingDataSimt_.tailBlockFactor = mBlockFactorTail_;
}

void MatrixDiagTiling::CalcScatterLowTilingData()
{
    int64_t coreNum = compileInfo_->coreNum;
    int64_t batchSize = fusedShape_[0];
    int64_t nSize = fusedShape_[1];
    int64_t ubEleNum =
        (xDtypeSize_ == 1) ? B8_MAX_SIZE / (nSize + 1) : (((ubSize_ - VL_LEN) / BASE_2) / (nSize + 1)) / xDtypeSize_;
    tilingDataScatter_.batchUbFactor = ubEleNum / nSize;
    tilingDataScatter_.batchUbCount = Ops::Base::CeilDiv(batchSize, tilingDataScatter_.batchUbFactor);
    if (tilingDataScatter_.batchUbCount < coreNum) {
        tilingDataScatter_.batchUbFactor = Ops::Base::CeilDiv(batchSize, coreNum);
        tilingDataScatter_.batchUbCount = Ops::Base::CeilDiv(batchSize, tilingDataScatter_.batchUbFactor);
    }
    tilingDataScatter_.batchUbTailFactor =
        batchSize - (tilingDataScatter_.batchUbCount - 1) * tilingDataScatter_.batchUbFactor;
    tilingDataScatter_.nSize = nSize;
    tilingDataScatter_.batchSize = batchSize;
    int64_t blockCount = std::min(tilingDataScatter_.batchUbCount, coreNum);
    tilingDataScatter_.realCoreNum = blockCount;
    tilingDataScatter_.blockFactor = Ops::Base::CeilDiv(tilingDataScatter_.batchUbCount, blockCount);
    if (blockCount > 0) {
        tilingDataScatter_.blockMainCount = (tilingDataScatter_.batchUbCount % blockCount == 0) ?
                                                blockCount :
                                                (tilingDataScatter_.batchUbCount % blockCount);
        tilingDataScatter_.blockTailFactor = tilingDataScatter_.batchUbCount / blockCount;
    }
}

void MatrixDiagTiling::CalcScatterLargeTilingData()
{
    int64_t coreNum = compileInfo_->coreNum;
    int64_t batchSize = fusedShape_[0];
    int64_t nSize = fusedShape_[1];
    int64_t ubEleNum = VL_LEN / xDtypeSize_;
    tilingDataLarge_.batchSize = batchSize;
    tilingDataLarge_.nSize = nSize;
    tilingDataLarge_.nUbFactor = ubEleNum;
    tilingDataLarge_.nUbCount = Ops::Base::CeilDiv(nSize, ubEleNum);
    tilingDataLarge_.nUbTailFactor = nSize % ubEleNum;

    int64_t blockCount = std::min(batchSize * tilingDataLarge_.nUbCount * tilingDataLarge_.nUbCount, coreNum);
    tilingDataLarge_.realCoreNum = blockCount;
    int64_t uBUnitCount = batchSize * tilingDataLarge_.nUbCount * tilingDataLarge_.nUbCount;
    tilingDataLarge_.blockFactor = Ops::Base::CeilDiv(uBUnitCount, blockCount);
    if (blockCount > 0) {
        tilingDataLarge_.blockMainCount = (uBUnitCount % blockCount == 0) ? blockCount : (uBUnitCount % blockCount);
        tilingDataLarge_.blockTailFactor = (uBUnitCount % blockCount == 0) ? 0 : uBUnitCount / blockCount;
    }
}

void MatrixDiagTiling::CalcScatterHighTilingData()
{
    int64_t coreNum = compileInfo_->coreNum;
    int64_t batchSize = fusedShape_[0];
    int64_t nSize = fusedShape_[1];
    int64_t halfUbEleNum = (xDtypeSize_ == 1) ? B8_MAX_SIZE : (ubSize_ - VL_LEN) / BASE_2 / xDtypeSize_;
    int64_t ubEleNum =
        (xDtypeSize_ == 1) ? std::sqrt(B8_MAX_SIZE) : (std::sqrt(1 + BASE_4 * halfUbEleNum) - 1) / BASE_2;
    tilingDataScatter_.batchUbFactor = ubEleNum / nSize;
    tilingDataScatter_.batchUbCount = Ops::Base::CeilDiv(batchSize, tilingDataScatter_.batchUbFactor);
    tilingDataScatter_.batchUbTailFactor =
        batchSize - (tilingDataScatter_.batchUbCount - 1) * tilingDataScatter_.batchUbFactor;
    tilingDataScatter_.nSize = nSize;
    tilingDataScatter_.realCoreNum = std::min(tilingDataScatter_.batchUbCount, coreNum);
    tilingDataScatter_.blockFactor =
        Ops::Base::CeilDiv(tilingDataScatter_.batchUbCount, tilingDataScatter_.realCoreNum);
    if (tilingDataScatter_.realCoreNum > 0) {
        tilingDataScatter_.blockMainCount = (tilingDataScatter_.batchUbCount % tilingDataScatter_.realCoreNum == 0) ?
                                                tilingDataScatter_.realCoreNum :
                                                (tilingDataScatter_.batchUbCount % tilingDataScatter_.realCoreNum);
        tilingDataScatter_.blockTailFactor = (tilingDataScatter_.batchUbCount % tilingDataScatter_.realCoreNum == 0) ?
                                                 0 :
                                                 tilingDataScatter_.batchUbCount / tilingDataScatter_.realCoreNum;
    }
}

void MatrixDiagTiling::SetBlockSplitInfo(int64_t batchBlockCnt, int64_t nBlockCnt, int64_t batchSize, int64_t nSize)
{
    // 采用尽量均匀的方式，如 14, 切分为4块，则切分为 4 4 3 3，有多个尾块，而不是 4 4 4 2这种.
    mBlockCount_ = batchBlockCnt;
    mBlockFactor_ = Ops::Base::CeilDiv(batchSize, mBlockCount_);
    mBlockFactorCount_ = (batchSize % mBlockCount_ == 0) ? mBlockCount_ : (batchSize % mBlockCount_);
    mBlockFactorTail_ = batchSize / mBlockCount_;

    nBlockCount_ = nBlockCnt;
    nBlockFactor_ = Ops::Base::CeilDiv(nSize, nBlockCount_);
    nBlockFactorCount_ = (nSize % nBlockCount_ == 0) ? nBlockCount_ : (nSize % nBlockCount_);
    nBlockFactorTail_ = nSize / nBlockCount_;

    realCoreNum_ = mBlockCount_ * nBlockCount_;

    OP_LOGI(
        context_->GetNodeName(),
        "Get block split TotalNum-BlockCnt-MainFactor-MainCnt-TailFactor, "
        "Batch:%ld %ld %ld %ld %ld, N:%ld %ld %ld %ld %ld",
        batchSize, mBlockCount_, mBlockFactor_, mBlockFactorCount_, mBlockFactorTail_, nSize, nBlockCount_,
        nBlockFactor_, nBlockFactorCount_, nBlockFactorTail_);
}

void MatrixDiagTiling::CalcPureCopyTilingData()
{
    int64_t coreNum = compileInfo_->coreNum;
    int64_t cores = std::min(coreNum, Ops::Base::CeilDiv(fusedShape_[0] * fusedShape_[1] * xDtypeSize_, BASE_512));
    SetBlockSplitInfo(cores, 1, fusedShape_[0], fusedShape_[1]);
    tilingDataPureCopy_.size = fusedShape_[0];
    tilingDataPureCopy_.ubSize = compileInfo_->ubSize;
    tilingDataPureCopy_.realCoreNum = cores;
    tilingDataPureCopy_.mainBlockCount = mBlockFactorCount_;
    tilingDataPureCopy_.mainBlockFactor = mBlockFactor_;
    tilingDataPureCopy_.tailBlockFactor = mBlockFactorTail_;
}

void MatrixDiagTiling::CalcTilingData()
{
    int64_t batchSize = fusedShape_[0];
    int64_t nSize = fusedShape_[1];
    ubSize_ = compileInfo_->ubSize;
    OP_LOGI(context_->GetNodeName(), "The ub size is: %ld.", ubSize_);
    if (nSize == 1) {
        tilingKey_ = ::MatrixDiagAsc::TILING_PURE_COPY;
        CalcPureCopyTilingData();
    } else if (batchSize * nSize <= SIMT_SIZE * compileInfo_->coreNum) {
        tilingKey_ = ::MatrixDiagAsc::TILING_SIMT;
        CalcSimtTilingData();
    } else if (nSize * xDtypeSize_ <= HALF_VL_LEN) {
        tilingKey_ = ::MatrixDiagAsc::TILING_SCATTER_LOW;
        CalcScatterLowTilingData();
    } else if (nSize * xDtypeSize_ <= VL_LEN) {
        tilingKey_ = ::MatrixDiagAsc::TILING_SCATTER_HIGH;
        CalcScatterHighTilingData();
    } else {
        tilingKey_ = ::MatrixDiagAsc::TILING_SCATTER_LARGE;
        CalcScatterLargeTilingData();
    }
}

void MatrixDiagTiling::FuseInputShape()
{
    int64_t nSize = 1;
    int64_t batchSize = 1;
    int64_t inputXDimNum = static_cast<int64_t>(inputShape_.GetDimNum());
    for (size_t i = 0; i < static_cast<size_t>(inputXDimNum - 1); i++) {
        batchSize *= inputShape_.GetDim(i);
    }

    nSize *= inputShape_.GetDim(inputXDimNum - 1);

    fusedShape_[0] = batchSize;
    fusedShape_[1] = nSize;

    OP_LOGI(context_->GetNodeName(), "After fused shape, batchSize:%ld nSize:%ld", fusedShape_[0], fusedShape_[1]);
}

ge::graphStatus MatrixDiagTiling::GetInputShapeAndType()
{
    auto xInput = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    auto xInputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    ge::DataType xDtype = xInputDesc->GetDataType();
    xDtypeSize_ = ge::GetSizeByDataType(xDtype);
    const gert::Shape& xInputShape = xInput->GetStorageShape();
    OP_CHECK_IF(
        xInputShape.GetDimNum() == 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "x", std::to_string(xInputShape.GetDimNum()).c_str(), "greater than 0"),
        return ge::GRAPH_FAILED);
    inputShape_ = xInputShape;
    FuseInputShape();
    OP_CHECK_IF(
        fusedShape_[0] == 0 || fusedShape_[1] == 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x", Ops::Base::ToString(inputShape_).c_str(),
            "fused batch size and n size must both be non-zero"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixDiagTiling::DoTiling()
{
    compileInfo_ = reinterpret_cast<const MatrixDiagCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);

    OP_CHECK_IF(
        GetInputShapeAndType() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Do tiling is failed!"),
        return ge::GRAPH_FAILED);
    CalcTilingData();
    return WriteTilingData();
}
} // namespace MatrixDiagAsc

static ge::graphStatus Tiling4MatrixDiag(gert::TilingContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Tiling4MatrixDiag", "context", "nullptr", "must not be null"),
        return ge::GRAPH_FAILED);

    MatrixDiagAsc::MatrixDiagTiling op(context);
    return op.DoTiling();
}

static ge::graphStatus TilingPrepare4MatrixDiagAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4MatrixDiagAscendC.");

    auto compileInfo = context->GetCompiledInfo<MatrixDiagAsc::MatrixDiagCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum < 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "core num", std::to_string(compileInfo->coreNum).c_str(),
            "must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<uint32_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize < 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "ub size", std::to_string(compileInfo->ubSize).c_str(),
            "must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);

    compileInfo->clSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        (compileInfo->clSize < 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "cache line size", std::to_string(compileInfo->clSize).c_str(),
            "must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        (compileInfo->blockSize < 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "block size", std::to_string(compileInfo->blockSize).c_str(),
            "must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4MatrixDiagAscendC.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4MatrixDiag(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TilingPrepare4MatrixDiag", "context", "nullptr", "must not be null"),
        return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<MatrixDiagAsc::MatrixDiagCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->isAscendC = Ops::Base::IsRegbaseSocVersion(context);
    if (compileInfo->isAscendC) {
        return TilingPrepare4MatrixDiagAscendC(context);
    }
    OP_LOGE("TilingPrepare4MatrixDiag", "AscendC TilingPrepare4MatrixDiag is failed.");
    return ge::GRAPH_FAILED;
}

// register tiling interface of the MatrixDiagAsc op.
IMPL_OP_OPTILING(MatrixDiag)
    .Tiling(Tiling4MatrixDiag)
    .TilingParse<MatrixDiagAsc::MatrixDiagCompileInfo>(TilingPrepare4MatrixDiag);
} // namespace optiling
