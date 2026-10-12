/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include "register/op_impl_registry.h"
#include "conversion/matrix_set_diag_v2/op_kernel/arch35/matrix_set_diag_v2_tilingdata.h"
#include "conversion/matrix_set_diag_v2/op_kernel/arch35/matrix_set_diag_v2_tilingkey.h"
#include "platform/platform_ascendc.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "op_host/util/const_util.h"
#include "log/log.h"
#include "matrix_set_diag_v2_tiling_arch35_base.h"

namespace optiling {
// BUFFER分割数量
static constexpr uint32_t BUFFER_NUM = 2;

static constexpr double SIMT_RATIO = 0.1;
static constexpr double SCATTER_RATIO = 1.2;
static constexpr uint32_t SIMT_DCACHE_SIZE = 32 * 1024U;
static constexpr uint32_t CUT_TAIL_X_MAX_SIZE = 32 * 1024U;
// UB内 scatter 操作的最大元素个数
static constexpr int32_t MAX_UB_SCATTER_ELEMENT_NUM = std::numeric_limits<int16_t>::max();

static constexpr int64_t MAX_UINT32_NUM = std::numeric_limits<uint32_t>::max();
static constexpr uint64_t MAX_DIAG_SIZE = 64 * 1024U;
// 单字节数据类型(dSize==1)的UB空间扩容倍数
static constexpr int32_t D_SIZE_EXPAND_TIMES = 2;
// 减半因子
static constexpr uint64_t HALVE_FACTOR = 2;
// 小数据类型(1~2字节)的最大字节数
static constexpr int64_t MAX_SMALL_DTYPE_SIZE = 2;
// SIMT 常量
static constexpr int64_t MAX_SHAPE_SIZE_FOR_SIMT = 1024;

static constexpr double MIN_USED_CORES_RATIO = 0.8;
static constexpr int64_t MIN_PER_UB_SIZE = 1024;
static constexpr int64_t MIN_PER_UB_SIZE_V1 = 4096;
static constexpr int32_t MAX_UB_SCATTER_ELEMENT_NUM_V1 = std::numeric_limits<uint16_t>::max();

void MatrixSetDiagTilingBase::CalcInputInfo()
{
    if (inputInfo_.dSize == 1) {
        dSizeExpand_ = D_SIZE_EXPAND_TIMES;
    }
    diagDataSize_ = inputInfo_.diagNum * inputInfo_.maxDiagLen;
    tailAxisDataSize_ = inputInfo_.xColNum * inputInfo_.xRowNum;
}

ge::graphStatus MatrixSetDiagTilingBase::DoTiling()
{
    CalcInputInfo();

    auto ret = GetSocInfo();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling GetSocInfo failed"), return ge::GRAPH_FAILED);

    ret = DoOpTiling();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoOpTiling failed"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    if (way_ == TPL_WAY_V1) {
        tilingKey = GET_TPL_TILING_KEY(static_cast<uint8_t>(TPL_WAY_V1), false, false, isCutTail_);
        OP_LOGI(context_->GetNodeName(), "tilingKey is %lu, way %d, isCutTail %d", tilingKey, way_, isCutTail_);
    } else {
        tilingKey = GET_TPL_TILING_KEY(way_, isVLFullLoad_, isBigShape_, isCutTail_);
        OP_LOGI(context_->GetNodeName(), "tilingKey is %lu, way %d, isVLFullLoad_ %d, isBigShape_ %d, isCutTail %d",
                tilingKey, way_, isVLFullLoad_, isBigShape_, isCutTail_);
    }

    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum_);

    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline T MatrixSetDiagTilingBase::AlignBlock(T elementCount)
{
    return Ops::Base::CeilAlign(elementCount, static_cast<T>(ubBlockElements_));
}

void MatrixSetDiagTilingBase::ShowCutTailTilingData()
{
    auto tilingData = context_->GetTilingData<MSDV2CutTailTilingData>();

    // 打印基础信息
    OP_LOGI(context_, "MSDV2CutTailTilingData:");
    OP_LOGI(context_, "\tinput: coreNum %u, mergeDimSize %lu, xRowNum %lu, xColNum %lu", tilingData->input.coreNum,
            tilingData->input.mergeDimSize, tilingData->input.xRowNum, tilingData->input.xColNum);
    OP_LOGI(context_, "\tinput: diagNum %lu, maxDiagLen %u, k0 %d, k1 %d", tilingData->input.diagNum,
            tilingData->input.maxDiagLen, tilingData->input.k0, tilingData->input.k1);
    OP_LOGI(context_, "\txRowFactor %lu, xColFactor %lu, totalCntPerCore %lu", tilingData->xRowFactor,
            tilingData->xColFactor, tilingData->totalCntPerCore);
}

void MatrixSetDiagTilingBase::ShowNoCutTailTilingData()
{
    OP_LOGI(context_, "way_ %d,isVLFullLoad_ %d", way_, isVLFullLoad_);
    auto tilingData = context_->GetTilingData<MSDV2NoCutTailTilingData>();

    // 打印基础信息
    OP_LOGI(context_, "MSDV2NoCutTailTilingData:");
    OP_LOGI(context_, "\tinput: coreNum %u, mergeDimSize %lu, xRowNum %lu, xColNum %lu", tilingData->input.coreNum,
            tilingData->input.mergeDimSize, tilingData->input.xRowNum, tilingData->input.xColNum);
    OP_LOGI(context_, "\tinput: diagNum %lu, maxDiagLen %u, k0 %d, k1 %d", tilingData->input.diagNum,
            tilingData->input.maxDiagLen, tilingData->input.k0, tilingData->input.k1);
    OP_LOGI(context_, "\tmergeDimNumPerCore %lu, ubFactor %lu", tilingData->mergeDimNumPerCore, tilingData->ubFactor);
}

void MatrixSetDiagTilingBase::FillsTilingData(MatrixSetDiagV2TilingData& tilingData)
{
    tilingData.coreNum = realCoreNum_;
    tilingData.mergeDimSize = inputInfo_.mergeDimSize;
    tilingData.xRowNum = inputInfo_.xRowNum;
    tilingData.xColNum = inputInfo_.xColNum;
    tilingData.diagNum = inputInfo_.diagNum;
    tilingData.maxDiagLen = inputInfo_.maxDiagLen;
    tilingData.k0 = inputInfo_.k0;
    tilingData.k1 = inputInfo_.k1;
}

ge::graphStatus MatrixSetDiagTilingBase::Tiling4CutTail()
{
    isCutTail_ = true;
    if (inputInfo_.k0 == inputInfo_.k1 && inputInfo_.k0 == 0) {
        way_ = TPL_WAY_V1;
        return Tiling4CutW();
    }
    isBigShape_ = diagDataSize_ > MAX_UINT32_NUM;

    // 可用UB大小
    OP_CHECK_IF((ubSize_ < SIMT_DCACHE_SIZE), OP_LOGE(context_, "ub size invalid"), return ge::GRAPH_FAILED);

    auto ret = context_->SetLocalMemorySize(ubSize_ - SIMT_DCACHE_SIZE);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "set local memory size failed."), return ret);

    uint32_t validBufSize = (ubSize_ - SIMT_DCACHE_SIZE) / BUFFER_NUM;
    uint64_t ubMaxDiagSize = std::min(
        static_cast<uint64_t>(AlignBlock(validBufSize / 3 * 2 / inputInfo_.dSize)) * inputInfo_.dSize, MAX_DIAG_SIZE);
    uint64_t ubMaxInputxSize = AlignBlock(ubMaxDiagSize / 2 / inputInfo_.dSize) * inputInfo_.dSize;

    if (AlignBlock(diagDataSize_) * inputInfo_.dSize < ubMaxDiagSize) {
        ubMaxDiagSize = AlignBlock(diagDataSize_) * inputInfo_.dSize;
        ubMaxInputxSize = validBufSize - ubMaxDiagSize;
    }
    ubMaxInputxSize = std::min(static_cast<uint32_t>(ubMaxInputxSize), CUT_TAIL_X_MAX_SIZE);
    auto tilingData = context_->GetTilingData<MSDV2CutTailTilingData>();
    CalculateCutTailTilingParams(ubMaxInputxSize, tilingData);

    if (realCoreNum_ <= coreNum_ / HALVE_FACTOR) {
        CalculateCutTailTilingParams(ubMaxInputxSize / HALVE_FACTOR, tilingData);
    }

    // 打印
    ShowCutTailTilingData();
    return ge::GRAPH_SUCCESS;
}

void MatrixSetDiagTilingBase::CalculateCutTailTilingParams(uint64_t ubMaxInputxSize, MSDV2CutTailTilingData* tilingData)
{
    uint64_t xNum = ubMaxInputxSize / inputInfo_.dSize;
    tilingData->xRowFactor = xNum > inputInfo_.xColNum ?
                                 std::min(Ops::Base::FloorDiv(xNum, inputInfo_.xColNum), inputInfo_.xRowNum) :
                                 1;
    tilingData->xColFactor = xNum > inputInfo_.xColNum ? inputInfo_.xColNum : xNum;
    uint64_t totalCount = inputInfo_.mergeDimSize *
                          Ops::Base::CeilDiv(inputInfo_.xRowNum, static_cast<uint64_t>(tilingData->xRowFactor)) *
                          Ops::Base::CeilDiv(inputInfo_.xColNum, static_cast<uint64_t>(tilingData->xColFactor));
    uint64_t perCoreCount = Ops::Base::CeilDiv(totalCount, static_cast<uint64_t>(coreNum_));
    realCoreNum_ = Ops::Base::CeilDiv(totalCount, perCoreCount);
    FillsTilingData(tilingData->input);
    tilingData->totalCntPerCore = perCoreCount;
}

ge::graphStatus MatrixSetDiagTilingBase::Tiling4NoCutTail()
{
    uint64_t additionTileSize = DetermineWayAndGetAdditionTileSize();
    uint64_t validBufSize = 0;

    auto ret = CalculateValidBufSize(additionTileSize, validBufSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    CalculateUbFactorAndCheck(validBufSize);
    if (ubFactor_ <= 0) {
        way_ = TPL_WAY_DEFAULT;
        isVLFullLoad_ = false;
        return Tiling4CutTail();
    }
    uint64_t totalTailSize = (tailAxisDataSize_ + diagDataSize_) * inputInfo_.dSize;
    uint64_t totalCount = Ops::Base::CeilDiv(inputInfo_.mergeDimSize, ubFactor_);
    uint64_t perCoreCount = Ops::Base::CeilDiv(totalCount, static_cast<uint64_t>(coreNum_));
    realCoreNum_ = Ops::Base::CeilDiv(totalCount, perCoreCount);
    if (static_cast<double>(realCoreNum_) / static_cast<double>(coreNum_) < MIN_USED_CORES_RATIO &&
        totalTailSize * ubFactor_ > MIN_PER_UB_SIZE) {
        GetOptimizeTilingNoCutTail();
        if (ubFactor_ <= 0) {
            way_ = TPL_WAY_DEFAULT;
            isVLFullLoad_ = false;
            return Tiling4CutTail();
        }
        totalCount = Ops::Base::CeilDiv(inputInfo_.mergeDimSize, ubFactor_);
        perCoreCount = Ops::Base::CeilDiv(ubTotalCount_, static_cast<uint64_t>(realCoreNum_));
    }
    ubTotalCount_ = totalCount;
    return FillNoCutTailTilingData();
}

uint64_t MatrixSetDiagTilingBase::DetermineWayAndGetAdditionTileSize()
{
    double ratio = static_cast<double>(diagDataSize_) / static_cast<double>(tailAxisDataSize_);
    uint64_t additionTileSize = 0;

    if (ratio >= SCATTER_RATIO) {
        way_ = TPL_WAY_GATHER;
        if (AlignBlock(tailAxisDataSize_) * inputInfo_.dSize * dSizeExpand_ <= vectorSize_) {
            isVLFullLoad_ = true;
        }
        additionTileSize = isVLFullLoad_ ? vectorSize_ :
                                           AlignBlock(tailAxisDataSize_) * inputInfo_.dSize * dSizeExpand_;
    } else if (ratio >= SIMT_RATIO) {
        way_ = TPL_WAY_SCATTER;
        if (AlignBlock(diagDataSize_) * inputInfo_.dSize * dSizeExpand_ <= vectorSize_) {
            isVLFullLoad_ = true;
        }
        additionTileSize = isVLFullLoad_ ? vectorSize_ : AlignBlock(diagDataSize_) * inputInfo_.dSize * dSizeExpand_;
    } else {
        way_ = TPL_WAY_SIMT;
    }

    return additionTileSize;
}

ge::graphStatus MatrixSetDiagTilingBase::CalculateValidBufSize(uint64_t additionTileSize, uint64_t& validBufSize)
{
    if (way_ == TPL_WAY_SIMT) {
        OP_CHECK_IF((ubSize_ < SIMT_DCACHE_SIZE), OP_LOGE(context_, "ub size invalid"), return ge::GRAPH_FAILED);
        auto ret = context_->SetLocalMemorySize(ubSize_ - SIMT_DCACHE_SIZE);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "set local memory size failed."), return ret);
        validBufSize = (ubSize_ - SIMT_DCACHE_SIZE) / BUFFER_NUM;
    } else {
        validBufSize = (ubSize_ - additionTileSize) / BUFFER_NUM;
    }
    return ge::GRAPH_SUCCESS;
}

void MatrixSetDiagTilingBase::CalculateUbFactorAndCheck(uint64_t validBufSize)
{
    uint64_t totalTailSize = (tailAxisDataSize_ + diagDataSize_) * inputInfo_.dSize;
    OP_LOGI(context_, "\ttotalTailSize %lu, validBufSize %lu", totalTailSize, validBufSize);
    if (totalTailSize == 0) {
        ubFactor_ = 1;
        return;
    }

    uint64_t ubComputeBufSize = validBufSize >= totalTailSize * inputInfo_.mergeDimSize ?
                                    totalTailSize * inputInfo_.mergeDimSize :
                                    validBufSize;
    ubFactor_ = ubComputeBufSize / totalTailSize;

    if (inputInfo_.dSize <= MAX_SMALL_DTYPE_SIZE && way_ != TPL_WAY_SIMT) {
        uint64_t tailFactor = (tailAxisDataSize_ > diagDataSize_) ? tailAxisDataSize_ : diagDataSize_;
        ubFactor_ = ubFactor_ * tailFactor < MAX_UB_SCATTER_ELEMENT_NUM ? ubFactor_ :
                                                                          MAX_UB_SCATTER_ELEMENT_NUM / tailFactor;
    }
}

ge::graphStatus MatrixSetDiagTilingBase::FillNoCutTailTilingData()
{
    auto tilingData = context_->GetTilingData<MSDV2NoCutTailTilingData>();
    uint64_t totalCount = Ops::Base::CeilDiv(inputInfo_.mergeDimSize, ubFactor_);
    uint64_t perCoreCount = Ops::Base::CeilDiv(totalCount, static_cast<uint64_t>(realCoreNum_));

    FillsTilingData(tilingData->input);
    tilingData->ubFactor = ubFactor_;
    tilingData->mergeDimNumPerCore = perCoreCount;

    ShowNoCutTailTilingData();
    return ge::GRAPH_SUCCESS;
}

void MatrixSetDiagTilingBase::GetOptimizeTilingNoCutTail()
{
    uint32_t startCoreNum = realCoreNum_ + 1;
    uint64_t curFactor = ubFactor_;
    for (uint32_t i = startCoreNum; i <= coreNum_; i++) {
        if (inputInfo_.mergeDimSize < i) {
            ubFactor_ = 0;
            break;
        }
        curFactor = Ops::Base::CeilDiv(inputInfo_.mergeDimSize, static_cast<uint64_t>(i));
        if (curFactor < ubFactor_) {
            uint64_t sizeTaken = AlignBlock(curFactor * (tailAxisDataSize_ + diagDataSize_) * inputInfo_.dSize);
            if (sizeTaken <= MIN_PER_UB_SIZE) {
                break;
            } else {
                ubFactor_ = curFactor;
                realCoreNum_ = i;
            }
        }
    }
}

ge::graphStatus MatrixSetDiagTilingBase::GetSocInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    realCoreNum_ = coreNum_;
    OP_CHECK_IF((coreNum_ == 0U), OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF((ubSize_ == 0U), OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ == 0U), OP_LOGE(context_, "Failed to get ub block size."), return ge::GRAPH_FAILED);
    ubBlockElements_ = ubBlockSize_ / inputInfo_.dSize;
    vectorSize_ = static_cast<uint64_t>(Ops::Base::GetVRegSize(context_));
    OP_CHECK_IF(vectorSize_ == 0U, OP_LOGE(context_, "Failed to get vector size."), return ge::GRAPH_FAILED);
    OP_LOGI(context_, "soc info: ubSize %lu, coreNum %u, ubBlockSize %lu ", ubSize_, coreNum_, ubBlockSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixSetDiagTilingBase::DoOpTiling()
{
    uint64_t totalTailSize = (AlignBlock(tailAxisDataSize_) + AlignBlock(diagDataSize_)) * inputInfo_.dSize;
    bufferSize_ = ubSize_ / BUFFER_NUM;
    OP_LOGI(context_, "bufferSize_ %lu, totalTailSize %lu, tailAxisDataSize_ %d, diagDataSize_ %d, inputInfo_.dSize %d",
            bufferSize_, totalTailSize, tailAxisDataSize_, diagDataSize_, inputInfo_.dSize);
    if (totalTailSize >= bufferSize_ ||
        (inputInfo_.dSize <= MAX_SMALL_DTYPE_SIZE &&
         (tailAxisDataSize_ >= MAX_UB_SCATTER_ELEMENT_NUM || diagDataSize_ >= MAX_UB_SCATTER_ELEMENT_NUM))) {
        return Tiling4CutTail();
    } else {
        return Tiling4NoCutTail();
    }
}

ge::graphStatus MatrixSetDiagTilingBase::Tiling4CutW()
{
    isCutTail_ = true;
    CalUbFactor();
    OP_CHECK_IF((ubFactor_ == 0U), OP_LOGE(context_, "ubFactor is 0"), return ge::GRAPH_FAILED);
    if (inputInfo_.dSize <= MAX_SMALL_DTYPE_SIZE) {
        ubFactor_ = ubFactor_ < MAX_UB_SCATTER_ELEMENT_NUM_V1 ? ubFactor_ : MAX_UB_SCATTER_ELEMENT_NUM_V1;
    }
    ubPerTail_ = Ops::Base::CeilDiv(tailAxisDataSize_, ubFactor_);
    ubFactor_ = Ops::Base::CeilDiv(tailAxisDataSize_, ubPerTail_);
    ubTotalCount_ = ubPerTail_ * inputInfo_.mergeDimSize;
    realCoreNum_ = ubTotalCount_ > coreNum_ ? coreNum_ : static_cast<uint32_t>(ubTotalCount_);
    ubPerCount_ = Ops::Base::CeilDiv(ubTotalCount_, static_cast<uint64_t>(realCoreNum_));
    ShowTilingData();
    GetOptimizeTiling();
    auto tilingData = context_->GetTilingData<MatrixSetDiagTilingData>();
    FillsTilingDataV1(*tilingData);
    return ge::GRAPH_SUCCESS;
}

void MatrixSetDiagTilingBase::CalUbFactor()
{
    uint64_t validBufSize = bufferSize_ - ubBlockSize_ * 2;
    if (inputInfo_.xColNum * inputInfo_.dSize >= bufferSize_) {
        ubFactor_ = validBufSize / inputInfo_.dSize;
    } else {
        uint64_t diagStride = inputInfo_.xColNum + 1;
        ubFactor_ = (validBufSize / inputInfo_.dSize * diagStride) / (diagStride + 1);
        while (ubFactor_ > 0 &&
               AlignBlock(ubFactor_) + AlignBlock(ubFactor_ / diagStride + 1) > bufferSize_ / inputInfo_.dSize) {
            ubFactor_ = ubFactor_ - 1;
        }
    }
}

void MatrixSetDiagTilingBase::GetOptimizeTiling()
{
    uint64_t curSize = CalSizeTaken(ubFactor_);
    uint64_t curFactor = ubFactor_;
    if (static_cast<double>(realCoreNum_) / static_cast<double>(coreNum_) >= MIN_USED_CORES_RATIO ||
        curSize <= MIN_PER_UB_SIZE_V1) {
        return;
    }
    uint32_t startCoreNum = realCoreNum_ + inputInfo_.mergeDimSize;
    for (uint32_t i = startCoreNum; i <= static_cast<uint32_t>(static_cast<double>(coreNum_) * MIN_USED_CORES_RATIO);) {
        curFactor = Ops::Base::CeilDiv(inputInfo_.mergeDimSize * tailAxisDataSize_, static_cast<uint64_t>(i));
        if (curFactor != ubFactor_) {
            if (CalSizeTaken(curFactor) <= MIN_PER_UB_SIZE_V1) {
                break;
            } else {
                ubFactor_ = curFactor;
                realCoreNum_ = i;
            }
        }
        i += inputInfo_.mergeDimSize;
    }
    ubPerTail_ = Ops::Base::CeilDiv(tailAxisDataSize_, ubFactor_);
    ubTotalCount_ = ubPerTail_ * inputInfo_.mergeDimSize;
    ubPerCount_ = Ops::Base::CeilDiv(ubTotalCount_, static_cast<uint64_t>(realCoreNum_));
    ShowTilingData();
}

uint64_t MatrixSetDiagTilingBase::CalSizeTaken(uint64_t factor)
{
    return (AlignBlock(factor) + AlignBlock(factor / (inputInfo_.xColNum + 1) + 1)) * inputInfo_.dSize;
}

void MatrixSetDiagTilingBase::ShowTilingData()
{
    OP_LOGI(context_, "ubFactor %lu, ubPerCore %lu, ubTotalCount %lu, ubPerTail %lu, isCutTail %d", ubFactor_,
            ubPerCount_, ubTotalCount_, ubPerTail_, isCutTail_);
}

void MatrixSetDiagTilingBase::FillsTilingDataV1(MatrixSetDiagTilingData& tilingData)
{
    tilingData.coreNum = realCoreNum_;
    tilingData.mergeDimSize = inputInfo_.mergeDimSize;
    tilingData.xRowNum = inputInfo_.xRowNum;
    tilingData.xColNum = inputInfo_.xColNum;
    tilingData.diagLen = diagDataSize_;
    tilingData.ubPerCore = ubPerCount_;
    tilingData.ubFactor = ubFactor_;
    tilingData.ubTotalCount = ubTotalCount_;
    tilingData.ubPerTail = ubPerTail_;
    tilingData.tailAxisDataSize = tailAxisDataSize_;
}
} // namespace optiling
