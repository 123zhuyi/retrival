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
 * \file dynamic_partition_tiling.cpp
 * \brief implemention of DynamicPartition tiling
 */
#include "dynamic_partition_tiling.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "log/log.h"

#include <cstring>

using namespace ge;

namespace optiling {
template <typename T>
inline static T* GetCompileInfoPtr(gert::TilingParseContext* context)
{
    return context->GetCompiledInfo<T>();
}

namespace DynPart {
static constexpr int64_t MC_TIMES = 2;
static constexpr int64_t SHAPE_BUF_SIZE = 96;
static constexpr uint32_t NUM_TWO = 2;
static constexpr uint32_t INT32_BYTES = 4;
static constexpr uint32_t UINT64_BYTES = 8;
static constexpr uint32_t W_THRESHOLD = 6;
static constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;
// arch35 MTE burst 粒度：每次 burst 传输 32 个 32 字节 block，即 1024 字节
static constexpr int64_t UB_BURST_IN_ONE_BLOCK = 32;

bool DynamicPartitionTiling::CheckInputs()
{
    auto xDimNum = xShape_.GetDimNum();
    auto partDimNum = partShape_.GetDimNum();

    OP_CHECK_IF(
        xDimNum < partDimNum,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "x and partitions", std::to_string(xDimNum) + " and " + std::to_string(partDimNum),
            "The dimNum of input x should be greater than or equal to the dimNum of input partitions"),
        return false);

    for (size_t i = 0; i < partDimNum; ++i) {
        OP_CHECK_IF(xShape_.GetDim(i) != partShape_.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), "x and partitions",
                        Ops::Base::ToString(xShape_) + " and " + Ops::Base::ToString(partShape_),
                        std::to_string(i) + " of x and " + std::to_string(i) + " of partitions must be equal"),
                    return false);
    }
    return true;
}

ge::graphStatus DynamicPartitionTiling::GetInputShapeAndType()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    xShape_ = xShapePtr->GetStorageShape();
    auto partShapePtr = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, partShapePtr);
    // Todo: when scalar, the process is different
    partShape_ = partShapePtr->GetStorageShape();

    auto xDescPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDescPtr);
    dtypeSize_ = GetSizeByDataType(xDescPtr->GetDataType());
    OP_CHECK_IF(dtypeSize_ == 0U, OP_LOGE(context_->GetNodeName(), "The data type size is zero!"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CheckInputs(), OP_LOGE(context_->GetNodeName(), "Get input shape and type is failed!"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void DynamicPartitionTiling::ReshapeInputShape()
{
    if (!xShape_.IsScalar()) {
        gert::Shape newPartShape;
        int64_t partShapeSize = 0;
        if (partShape_.IsScalar()) {
            partShapeSize = 1L;
        } else {
            partShapeSize = partShape_.GetShapeSize();
        }
        newPartShape.AppendDim(partShapeSize);
        gert::Shape newXShape;
        newXShape.AppendDim(partShapeSize);
        if (partShapeSize > 0) {
            newXShape.AppendDim(xShape_.GetShapeSize() / partShapeSize);
        }
        xShape_ = newXShape;
        partShape_ = newPartShape;
    }
}

void DynamicPartitionTiling::CalcTilingExtFirstOutDims()
{
    auto xDimNum = xShape_.GetDimNum();
    for (auto i = partShape_.GetDimNum(); i < xDimNum; ++i) {
        tilingData_.outDimsExtFirst[tilingData_.dimNumExtFirst++] = xShape_.GetDim(i);
    }
}

void DynamicPartitionTiling::CalcTilingMCHWSize()
{
    if (xShape_.IsScalar()) {
        tilingData_.usedCoreCnt = 1L;
        return;
    }

    int64_t clSize = static_cast<int64_t>(compileInfo_->clSize / dtypeSize_);
    int64_t coreNum = static_cast<int64_t>(compileInfo_->coreNum);
    if (xShape_.GetShapeSize() == 0L) {
        tilingData_.usedCoreCnt = 1L;
        if (!partShape_.IsScalar()) {
            auto pSize = partShape_.GetDim(0);
            auto pUnitCnt = Ops::Base::CeilDiv(pSize, clSize);
            tilingData_.usedCoreCnt = (pSize == 0) ?
                                          1L :
                                          Ops::Base::CeilDiv(pUnitCnt, Ops::Base::CeilDiv(pUnitCnt, coreNum));
            tilingData_.hMSize = Ops::Base::CeilDiv(pUnitCnt, tilingData_.usedCoreCnt) * clSize;
            tilingData_.hTSize = pSize - tilingData_.hMSize * (tilingData_.usedCoreCnt - 1);
        }
        return;
    }

    auto hSize = xShape_.GetDim(0);
    auto wSize = xShape_.GetDim(1);
    tilingData_.hOffset = wSize;
    auto minHSize = std::min(Ops::Base::CeilDiv(clSize, wSize), hSize);
    auto newHSize = Ops::Base::CeilDiv(hSize, minHSize);
    auto newWSize = Ops::Base::CeilDiv(wSize, clSize);

    auto hMCores = Ops::Base::CeilDiv(newHSize, Ops::Base::CeilDiv(newHSize, coreNum));
    auto wMCores = Ops::Base::CeilDiv(newWSize, Ops::Base::CeilDiv(newWSize, coreNum));
    if (wMCores * MC_TIMES < hMCores) {
        tilingData_.hMSize = Ops::Base::CeilDiv(newHSize, hMCores) * minHSize;
        tilingData_.hTSize = hSize - tilingData_.hMSize * (hMCores - 1);
        tilingData_.wMSize = wSize;
        tilingData_.wTSize = wSize;
        tilingData_.usedCoreCnt = hMCores;
        isHBlockAxis_ = true;
    } else {
        tilingData_.hMSize = hSize;
        tilingData_.hTSize = hSize;
        tilingData_.wMSize = Ops::Base::CeilDiv(newWSize, wMCores) * clSize;
        tilingData_.wTSize = wSize - tilingData_.wMSize * (wMCores - 1);
        tilingData_.usedCoreCnt = wMCores;
    }
}

void DynamicPartitionTiling::CalcTilingHWLpUnit()
{
    // move in W first
    if (xShape_.GetShapeSize() > 1L && partShape_.GetShapeSize() >= 1L) {
        // ping pong
        int64_t blockSize = static_cast<int64_t>(compileInfo_->blockSize);
        int64_t coreWSAlign_ = Ops::Base::CeilAlign(static_cast<int64_t>(coreWS_), blockSize);
        int64_t availUBSize = static_cast<int64_t>(compileInfo_->ubSize - coreWSAlign_ * NUM_TWO) / NUM_TWO;
        int64_t clSize = static_cast<int64_t>(compileInfo_->clSize / dtypeSize_);
        auto hMSize = tilingData_.hMSize;
        auto wMSize = tilingData_.wMSize;
        // first two xIn and xOut, second two partition and partition seq

        if (Ops::Base::CeilAlign(wMSize * dtypeSize_, blockSize) * NUM_TWO + blockSize * NUM_TWO >= availUBSize) {
            tilingData_.hLpUnit = blockSize / INT32_BYTES;
            tilingData_.wLpUnit = Ops::Base::CeilAlign(
                (availUBSize - blockSize * NUM_TWO) / NUM_TWO / tilingData_.hLpUnit, clSize);
            return;
        }
        if ((Ops::Base::CeilAlign(wMSize * dtypeSize_, blockSize) * hMSize * NUM_TWO +
             Ops::Base::CeilAlign(hMSize * INT32_BYTES, blockSize) * NUM_TWO) <= availUBSize) {
            tilingData_.hLpUnit = hMSize;
            tilingData_.wLpUnit = wMSize;
        } else {
            if (wMSize <= W_THRESHOLD) {
                while ((Ops::Base::CeilAlign(wMSize * hMSize * dtypeSize_, blockSize) * NUM_TWO +
                        Ops::Base::CeilAlign(wMSize * hMSize * INT32_BYTES, blockSize) * NUM_TWO) > availUBSize) {
                    --hMSize;
                }
            } else {
                while ((Ops::Base::CeilAlign(wMSize * dtypeSize_, blockSize) * hMSize * NUM_TWO +
                        Ops::Base::CeilAlign(hMSize * INT32_BYTES, blockSize) * NUM_TWO) > availUBSize) {
                    --hMSize;
                }
            }
            tilingData_.hLpUnit = hMSize;
            tilingData_.wLpUnit = wMSize;
        }
    }
}

void DynamicPartitionTiling::CalcTilingUB()
{
    tilingData_.totalUBSize = compileInfo_->ubSize;
    int64_t blockSize = static_cast<int64_t>(compileInfo_->blockSize);
    int64_t wLpUnitAlign_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_.wLpUnit * dtypeSize_), blockSize);
    tilingData_.dataUBSize = static_cast<uint32_t>(tilingData_.hLpUnit * wLpUnitAlign_);
    tilingData_.partUBSize = static_cast<uint32_t>(tilingData_.hLpUnit * INT32_BYTES);
}

void DynamicPartitionTiling::CalcTilingKey()
{
    // here partitions must be scalar
    if (xShape_.IsScalar()) {
        tilingData_.tilingKey = ::DynPart::TILING_XP_SCALAR;
        return;
    }

    if (xShape_.GetShapeSize() == 0) {
        if (partShape_.IsScalar() || partShape_.GetShapeSize() > 0) {
            tilingData_.tilingKey = ::DynPart::TILING_X_EMPTY;
        } else {
            tilingData_.tilingKey = ::DynPart::TILING_XP_EMPTY;
        }
        return;
    }

    if (tilingData_.numPartitions == 1) {
        tilingData_.tilingKey = ::DynPart::TILING_NUM_PART_ONE;
        return;
    }

    if (tilingData_.wMSize == tilingData_.wLpUnit) {
        if (isHBlockAxis_) {
            if (tilingData_.wMSize <= W_THRESHOLD) {
                tilingData_.tilingKey = ::DynPart::TILING_H_MC_WITH_SMALL_W;
            } else {
                tilingData_.tilingKey = ::DynPart::TILING_H_MC_UB_CAN_HOLD_SPLIT_W;
            }
        } else {
            tilingData_.tilingKey = ::DynPart::TILING_W_MC_UB_CAN_HOLD_SPLIT_W;
        }
        return;
    }

    if (tilingData_.wMSize != tilingData_.wLpUnit) {
        if (isHBlockAxis_) {
            tilingData_.tilingKey = ::DynPart::TILING_H_MC_UB_CANNOT_HOLD_SPLIT_W;
        } else {
            tilingData_.tilingKey = ::DynPart::TILING_W_MC_UB_CANNOT_HOLD_SPLIT_W;
        }
        return;
    }
}

ge::graphStatus DynamicPartitionTiling::GetAttrNumPartitions()
{
    auto ptrAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrAttrs);
    if (ptrAttrs->GetAttrNum() > 0) {
        const int32_t* ptrNumParts = ptrAttrs->GetAttrPointer<int32_t>(0);
        OP_CHECK_NULL_WITH_CONTEXT(context_, ptrNumParts);
        int32_t numParts = *ptrNumParts;
        OP_CHECK_IF(
            numParts < 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_partitions", std::to_string(numParts),
                                                  "The value of num_partitions cannot be less than 1"),
            return ge::GRAPH_FAILED);
        tilingData_.numPartitions = numParts;
        coreWS_ = std::min(tilingData_.numPartitions, ::DynPart::NUM_PARTITION_UNIT) * UINT64_BYTES;
    }
    return ge::GRAPH_SUCCESS;
}

void DynamicPartitionTiling::CalcTilingDataNumPartOne()
{
    int64_t blockSize = static_cast<int64_t>(compileInfo_->blockSize);
    int64_t elePerBlock = blockSize / static_cast<int64_t>(dtypeSize_);
    int64_t coreNum = static_cast<int64_t>(compileInfo_->coreNum);
    int64_t totalUB = static_cast<int64_t>(compileInfo_->ubSize);
    int64_t totalElements = xShape_.GetShapeSize();
    int64_t optBurstElems = UB_BURST_IN_ONE_BLOCK * elePerBlock; // 最优 DMA burst 粒度（元素数）

    numPartOneTilingData_.tilingKey = ::DynPart::TILING_NUM_PART_ONE;
    numPartOneTilingData_.totalElements = totalElements;
    numPartOneTilingData_.totalUBSize = compileInfo_->ubSize;
    numPartOneTilingData_.mainSize = std::max(
        Ops::Base::CeilAlign(Ops::Base::CeilDiv(totalElements, coreNum), elePerBlock), optBurstElems);
    numPartOneTilingData_.usedCoreCnt = Ops::Base::CeilDiv(totalElements, numPartOneTilingData_.mainSize);
    numPartOneTilingData_.tailSize = totalElements -
                                     numPartOneTilingData_.mainSize * (numPartOneTilingData_.usedCoreCnt - 1);

    // 预留 shapeBuf_ 的 UB 开销（96 字节，与 kernel InitBuffer(shapeBuf_, SHAPE_BUF_SIZE=96) 对齐）
    int64_t availUB = totalUB - SHAPE_BUF_SIZE;
    int64_t ubFactor = (availUB / 2 / blockSize) * elePerBlock;
    ubFactor = std::min(ubFactor, numPartOneTilingData_.mainSize);
    ubFactor = std::max(ubFactor, elePerBlock);
    numPartOneTilingData_.ubFactor = ubFactor;

    // 填充 y[0] shape 信息（供 kernel 写 yshape 输出）
    // 注意：此时 xShape_ 已被 ReshapeInputShape() 重塑为 [partSize, wSize]
    // dim0 = 原始 partitions 的元素总数，即重塑后 xShape_[0]
    numPartOneTilingData_.dim0 = xShape_.GetDim(0);
    numPartOneTilingData_.dimNumExtFirst = tilingData_.dimNumExtFirst;
    for (int64_t i = 0; i < tilingData_.dimNumExtFirst; ++i) {
        numPartOneTilingData_.outDimsExtFirst[i] = tilingData_.outDimsExtFirst[i];
    }
}

void DynamicPartitionTiling::CalcTilingData()
{
    CalcTilingExtFirstOutDims();
    ReshapeInputShape();
    CalcTilingMCHWSize();
    CalcTilingHWLpUnit();
    CalcTilingUB();
    CalcTilingKey();
    if (tilingData_.tilingKey == ::DynPart::TILING_NUM_PART_ONE) {
        CalcTilingDataNumPartOne();
    }
}

std::string DynamicPartitionTiling::PrintTilingData()
{
    std::string tdStr;
    tdStr += std::to_string(tilingData_.tilingKey) + ",";
    tdStr += std::to_string(tilingData_.usedCoreCnt) + ",";
    tdStr += std::to_string(tilingData_.dataUBSize) + ",";
    tdStr += std::to_string(tilingData_.partUBSize) + ",";
    tdStr += std::to_string(tilingData_.totalUBSize) + ",";
    tdStr += std::to_string(tilingData_.numPartitions) + ",";
    tdStr += std::to_string(tilingData_.hMSize) + ",";
    tdStr += std::to_string(tilingData_.hTSize) + ",";
    tdStr += std::to_string(tilingData_.hLpUnit) + ",";
    tdStr += std::to_string(tilingData_.hOffset) + ",";
    tdStr += std::to_string(tilingData_.wMSize) + ",";
    tdStr += std::to_string(tilingData_.wTSize) + ",";
    tdStr += std::to_string(tilingData_.wLpUnit) + ",";
    tdStr += std::to_string(tilingData_.dimNumExtFirst) + ",";
    tdStr += "output dims except first:";
    for (int64_t i = 0; i < tilingData_.dimNumExtFirst; ++i) {
        tdStr += std::to_string(tilingData_.outDimsExtFirst[i]) + " ";
    }
    return tdStr;
}

ge::graphStatus DynamicPartitionTiling::WriteTilingData()
{
    OP_CHECK_IF(context_->SetTilingKey(tilingData_.tilingKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set tiling key is failed!"), return ge::GRAPH_FAILED);

    if (tilingData_.tilingKey == ::DynPart::TILING_NUM_PART_ONE) {
        OP_CHECK_IF(
            context_->SetBlockDim(static_cast<uint32_t>(numPartOneTilingData_.usedCoreCnt)) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "Set used core size is failed!"), return ge::GRAPH_FAILED);
        auto ptrTD = context_->GetRawTilingData();
        OP_CHECK_NULL_WITH_CONTEXT(context_, ptrTD);
        auto capSize = ptrTD->GetCapacity();
        void* ptrData = ptrTD->GetData();
        OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
        OP_CHECK_IF(memcpy_s(ptrData, capSize, &numPartOneTilingData_, sizeof(numPartOneTilingData_)) != 0,
                    OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
        ptrTD->SetDataSize(sizeof(numPartOneTilingData_));
        size_t* ptrWS = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, ptrWS);
        ptrWS[0] = SYS_WORKSPACE_SIZE;
        OP_LOGI(
            context_->GetNodeName(),
            "The tiling data (NumPartOne): key=%lu coreCnt=%ld totalElem=%ld mainSize=%ld tailSize=%ld ubFactor=%ld",
            numPartOneTilingData_.tilingKey, numPartOneTilingData_.usedCoreCnt, numPartOneTilingData_.totalElements,
            numPartOneTilingData_.mainSize, numPartOneTilingData_.tailSize, numPartOneTilingData_.ubFactor);
        return ge::GRAPH_SUCCESS;
    }

    if (tilingData_.tilingKey == ::DynPart::TILING_X_EMPTY ||
        tilingData_.tilingKey == ::DynPart::TILING_H_MC_UB_CAN_HOLD_SPLIT_W ||
        tilingData_.tilingKey == ::DynPart::TILING_H_MC_UB_CANNOT_HOLD_SPLIT_W ||
        tilingData_.tilingKey == ::DynPart::TILING_H_MC_WITH_SMALL_W) {
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to set ScheduleMode!"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(context_->SetBlockDim(static_cast<uint32_t>(tilingData_.usedCoreCnt)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set used core size is failed!"), return ge::GRAPH_FAILED);

    auto ptrTD = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrTD);
    auto capSize = ptrTD->GetCapacity();
    void* ptrData = ptrTD->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    ptrTD->SetDataSize(sizeof(tilingData_));

    size_t usrWorkspaceSize = 0;
    if (isHBlockAxis_ || tilingData_.tilingKey == ::DynPart::TILING_X_EMPTY) {
        usrWorkspaceSize = static_cast<size_t>(tilingData_.usedCoreCnt * coreWS_);
    }
    size_t totalWorkspaceSize = usrWorkspaceSize + SYS_WORKSPACE_SIZE;
    size_t* ptrWS = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrWS);
    ptrWS[0] = totalWorkspaceSize;

    OP_LOGI(context_->GetNodeName(), "The tiling data is: %s", PrintTilingData().c_str());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicPartitionTiling::DoTiling()
{
    compileInfo_ = context_->GetCompileInfo<DynamicPartitionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);

    OP_CHECK_IF(GetInputShapeAndType() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Do tiling is failed!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrNumPartitions() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Do tiling is failed!"),
                return ge::GRAPH_FAILED);
    CalcTilingData();
    return WriteTilingData();
}
} // namespace DynPart

static ge::graphStatus Tiling4DynamicPartition(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Tiling4DynamicPartition", "The context is nullptr!"),
                return ge::GRAPH_FAILED);

    DynPart::DynamicPartitionTiling op(context);
    return op.DoTiling();
}

static ge::graphStatus TilingPrepare4DynamicPartitionAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4DynamicPartitionAscendC.");

    auto compileInfo = GetCompileInfoPtr<DynPart::DynamicPartitionCompileInfo>(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum < 1),
                OP_LOGE(context->GetNodeName(), "The core num is invalid, %u.", compileInfo->coreNum),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<uint32_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize < 1),
                OP_LOGE(context->GetNodeName(), "The ub size is invalid, %u.", compileInfo->ubSize),
                return ge::GRAPH_FAILED);

    compileInfo->clSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF((compileInfo->clSize < 1),
                OP_LOGE(context->GetNodeName(), "The cache line size is invalid, %u.", compileInfo->clSize),
                return ge::GRAPH_FAILED);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF((compileInfo->blockSize < 1),
                OP_LOGE(context->GetNodeName(), "The block size is invalid, %u.", compileInfo->blockSize),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4DynamicPartitionAscendC.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the DynamicPartition op.
IMPL_OP_OPTILING(DynamicPartition)
    .Tiling(Tiling4DynamicPartition)
    .TilingParse<DynPart::DynamicPartitionCompileInfo>(TilingPrepare4DynamicPartitionAscendC);
} // namespace optiling
