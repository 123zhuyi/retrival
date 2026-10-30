/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sort_tiling_arch35.cpp
 * \brief sort ac tiling impl
 */
#include "sort_tiling_arch35.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/sort_tiling_data.h"
#include "../../op_kernel/arch35/sort_tiling_key.h"
#include "sort_tiling_common.h"

namespace optiling {
// =============================================================================
// Parameter validation
// =============================================================================
ge::graphStatus CheckInputAndOutput(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(SIMT_UB),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                                      "The value of ubSize must be greater than SIMT_UB."),
                return ge::GRAPH_FAILED);
    sortTileInfo.blockUbSize = Ops::Base::GetUbBlockSize(context);
    OP_LOGI(context->GetNodeName(), "ubSize is %ld, blockUbSize %u", ubSize, sortTileInfo.blockUbSize);
    sortTileInfo.ubSize = static_cast<uint32_t>(ubSize);
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(inputShapePtr->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    const gert::Shape& outShape = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());
    auto yStorage1 = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage1);
    const gert::Shape& outShape1 = Ops::Base::EnsureNotScalar(yStorage1->GetStorageShape());
    OP_CHECK_IF(
        inputShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "x, y1",
            (std::to_string(inputShape.GetShapeSize()) + ", " + std::to_string(outShape.GetShapeSize())).c_str(),
            "The values of shape sizes of x and y1 must be positive."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outShape != outShape1 || outShape != inputShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y1, y2",
                                               (Ops::Base::ToString(inputShape) + ", " + Ops::Base::ToString(outShape) +
                                                ", " + Ops::Base::ToString(outShape1))
                                                   .c_str(),
                                               "The shapes of x, y1, and y2 must be the same."),
        return ge::GRAPH_FAILED);
    int32_t xDimNum = inputShape.GetDimNum();
    sortTileInfo.rank = xDimNum;
    sortTileInfo.sortAxis = xDimNum - 1;
    ComputeAxisDimProducts(inputShape, sortTileInfo.sortAxis, sortTileInfo);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSortOutputDtypes(gert::TilingContext* context, SortKthTileInfo& sortTileInfo,
                                      ge::DataType dataType)
{
    auto outDescPtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr);
    auto y2DType = outDescPtr->GetDataType();
    auto outDescPtr0 = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr0);
    auto y1DType = outDescPtr0->GetDataType();
    OP_CHECK_IF(
        (y2DType != ge::DT_INT64) && (y2DType != ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y2", Ops::Base::ToString(y2DType).c_str(), "INT32 or INT64"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(y1DType != dataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x, y1",
                    (Ops::Base::ToString(dataType) + ", " + Ops::Base::ToString(y1DType)).c_str(),
                    "The dtypes of x and y1 must be the same."),
                return ge::GRAPH_FAILED);
    ge::TypeUtils::GetDataTypeLength(y2DType, sortTileInfo.y2DtypeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComputeNonLastAxisLayout(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, int32_t sortAxis)
{
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(inputShapePtr->GetStorageShape());
    ComputeAxisDimProducts(inputShape, sortAxis, sortTileInfo);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortCheckParams(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    OP_CHECK_IF(CheckInputAndOutput(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CheckInputAndOutput", "GRAPH_FAILED",
                                                      "The value of CheckInputAndOutput must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    auto inputDescPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
    ge::DataType dataType = inputDescPtr->GetDataType();
    sortTileInfo.dataType = dataType;
    OP_CHECK_IF(
        !ge::TypeUtils::GetDataTypeLength(dataType, sortTileInfo.dtypeSize),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                  "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16, BF16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckSortOutputDtypes(context, sortTileInfo, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CheckSortOutputDtypes", "GRAPH_FAILED",
                                                      "The value of CheckSortOutputDtypes must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isDescending = attrs->GetAttrPointer<bool>(1);
    const int64_t* sortAxisPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, isDescending);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortAxisPtr);
    int32_t sortAxis = static_cast<int32_t>(*sortAxisPtr);
    sortAxis = sortAxis < 0 ? (sortAxis + static_cast<int32_t>(sortTileInfo.rank)) : sortAxis;
    OP_CHECK_IF(sortAxis < 0 || sortAxis >= static_cast<int32_t>(sortTileInfo.rank),
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "axis", std::to_string(sortAxis).c_str(),
                                          "range [-dimNum, dimNum - 1)"),
                return ge::GRAPH_FAILED);
    sortTileInfo.sortAxis = sortAxis;
    sortTileInfo.isNonLastAxis = (sortAxis != (static_cast<int32_t>(sortTileInfo.rank) - 1));
    if (sortTileInfo.isNonLastAxis) {
        OP_CHECK_IF(
            ComputeNonLastAxisLayout(context, sortTileInfo, sortAxis) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeNonLastAxisLayout", "GRAPH_FAILED",
                                                  "The value of ComputeNonLastAxisLayout must be "
                                                  "GRAPH_SUCCESS."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// UB computation and layout helpers
// =============================================================================
void SetSortTmpSize(ge::DataType dataType, uint32_t tileData, bool isDescend, SortKthTileInfo& sortTileInfo)
{
    int64_t realLen = std::min(sortTileInfo.lastAxis, static_cast<int64_t>(tileData));
    std::vector<int64_t> shapeVec = {realLen};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTiling", "api of sort shape is %ld, maxUb is %u", realLen, maxValue);
    sortTileInfo.tmpUbSize = maxValue;
    return;
}

// =============================================================================
// Non-last axis helpers
// =============================================================================
static bool TryGetSortNonLastTileCount(const SortKthTileInfo& info, uint32_t innerChunk, uint32_t& innerLoopNum,
                                       uint64_t& tileCount)
{
    if (innerChunk == 0U || info.innerSize <= 0 || info.outerSize <= 0) {
        return false;
    }
    uint64_t innerLoopNum64 = Ops::Base::CeilDiv(static_cast<uint64_t>(info.innerSize),
                                                 static_cast<uint64_t>(innerChunk));
    if (innerLoopNum64 == 0U || innerLoopNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    tileCount = static_cast<uint64_t>(info.outerSize) * innerLoopNum64;
    innerLoopNum = static_cast<uint32_t>(innerLoopNum64);
    return tileCount > 0U;
}

// Clamp batchSize to innerSize, then compute total tile count across all outer batches.
// Returns false on overflow or zero tile count.
bool ComputeSortNonLastBatchPlan(const SortKthTileInfo& sortTileInfo, uint32_t& batchSize, uint32_t& innerLoopNum,
                                 uint32_t& batchNum)
{
    batchSize = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(batchSize), static_cast<uint64_t>(sortTileInfo.innerSize)));
    if (!ComputeNonLastBatchNum(sortTileInfo.outerSize, sortTileInfo.innerSize, batchSize, batchNum)) {
        return false;
    }
    uint64_t tileCount64 = 0;
    if (!TryGetSortNonLastTileCount(sortTileInfo, batchSize, innerLoopNum, tileCount64)) {
        return false;
    }
    if (tileCount64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    batchNum = static_cast<uint32_t>(tileCount64);
    return batchNum > 0;
}

bool IsRadixSortOneCore(SortKthTileInfo& sortTileInfo)
{
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        return false;
    }
    uint32_t xUbSize = 0;
    uint32_t y2UbSize = 0;
    if (!ComputeRadixOneCoreUbSizes(sortTileInfo.lastAxis, sortTileInfo.dtypeSize,
                                    static_cast<uint32_t>(sizeof(int32_t)), sortTileInfo.blockUbSize, xUbSize,
                                    y2UbSize)) {
        return false;
    }

    // Sort API writes uint32 indices first. For int64 output, reserve the other half for Cast result.
    uint32_t halfNum = y2UbSize / static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.y2DtypeSize == static_cast<uint32_t>(sizeof(int64_t))) {
        y2UbSize = y2UbSize * static_cast<uint32_t>(sizeof(int64_t) / sizeof(int32_t));
    }
    sortTileInfo.keyParams0 = xUbSize;
    sortTileInfo.keyParams1 = y2UbSize;
    sortTileInfo.keyParams2 = halfNum;
    sortTileInfo.keyParams3 = 1;
    // keyParams3 records whether the queues can use double buffer after reserving Sort tmp UB.
    int64_t oneBufferQueSize = static_cast<int64_t>(xUbSize) * 2 + static_cast<int64_t>(y2UbSize);
    int64_t remainUb = static_cast<int64_t>(sortTileInfo.ubSize) - oneBufferQueSize;
    if (remainUb <= static_cast<int64_t>(0)) {
        return false;
    }
    remainUb = (remainUb / static_cast<int64_t>(sortTileInfo.blockUbSize)) *
               static_cast<int64_t>(sortTileInfo.blockUbSize);
    SetSortTmpSize(sortTileInfo.dataType, static_cast<uint32_t>(sortTileInfo.lastAxis), sortTileInfo.isDescend,
                   sortTileInfo);
    int64_t tmpUb = static_cast<int64_t>(sortTileInfo.tmpUbSize);
    OP_LOGI("RadixSortTiling", "remainUb is %ld, tmpUb is %ld", remainUb, tmpUb);
    if (tmpUb > remainUb) {
        return false;
    }

    int64_t doubleBufferRemainUb = static_cast<int64_t>(sortTileInfo.ubSize) - oneBufferQueSize * DOUBLE_BUFFER_NUM;
    doubleBufferRemainUb = (doubleBufferRemainUb / static_cast<int64_t>(sortTileInfo.blockUbSize)) *
                           static_cast<int64_t>(sortTileInfo.blockUbSize);
    if (tmpUb <= doubleBufferRemainUb) {
        sortTileInfo.keyParams3 = DOUBLE_BUFFER_NUM;
    }
    OP_LOGI("RadixSortTiling", "radix one-core bufferNum is %u", sortTileInfo.keyParams3);
    return true;
}

// =============================================================================
// Individual strategy Set functions
// =============================================================================
bool IsAxisOneCopy(const SortKthTileInfo& sortTileInfo) { return sortTileInfo.lastAxis == static_cast<int64_t>(1); }

ge::graphStatus SetAxisOneCopyTiling(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    // double buffer
    uint64_t bytesPerElem = static_cast<uint64_t>(2) * (static_cast<uint64_t>(sortTileInfo.dtypeSize) +
                                                        static_cast<uint64_t>(sortTileInfo.y2DtypeSize));
    if (bytesPerElem == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                              Ops::Base::ToString(sortTileInfo.dataType).c_str(),
                                              "The dtype size of x must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t copyElemsPerLoop64 = static_cast<uint64_t>(sortTileInfo.ubSize) / bytesPerElem;
    if (copyElemsPerLoop64 == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "copyElemsPerLoop",
                                              std::to_string(copyElemsPerLoop64).c_str(),
                                              "The value of copyElemsPerLoop must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    if (copyElemsPerLoop64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "copyElemsPerLoop", std::to_string(copyElemsPerLoop64).c_str(),
            "The value of copyElemsPerLoop must be less than or equal to uint32 max.");
        return ge::GRAPH_FAILED;
    }
    uint32_t copyElemsPerLoop = static_cast<uint32_t>(copyElemsPerLoop64);
    uint64_t totalElems = static_cast<uint64_t>(sortTileInfo.unsortedDim) *
                          static_cast<uint64_t>(sortTileInfo.lastAxis);
    uint64_t loopTimes64 = (totalElems + copyElemsPerLoop64 - 1) / copyElemsPerLoop64;
    if (loopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "loopTimes", std::to_string(loopTimes64).c_str(),
                                              "The value of loopTimes must be less than or equal to uint32 max.");
        return ge::GRAPH_FAILED;
    }
    uint32_t loopTimes = static_cast<uint32_t>(loopTimes64);

    uint32_t coreNumNeed = std::min(sortTileInfo.maxCoreNum, loopTimes);
    sortTileInfo.numTileDataSize = copyElemsPerLoop;
    sortTileInfo.keyParams0 = copyElemsPerLoop;
    sortTileInfo.keyParams1 = loopTimes;
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.unsortedDimParallel = coreNumNeed;
    sortTileInfo.lastDimTileNum = 1;
    sortTileInfo.lastDimNeedCore = 1;
    sortTileInfo.sortLoopTimes = Ops::Base::CeilDiv(static_cast<int64_t>(loopTimes), static_cast<int64_t>(coreNumNeed));
    sortTileInfo.tmpUbSize = 0;

    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    OP_LOGI("AxisOneCopyTiling", "totalElems %lu, copyElemsPerLoop %u, loopTimes %u, coreNumNeed %u", totalElems,
            sortTileInfo.keyParams0, sortTileInfo.keyParams1, coreNumNeed);
    return ge::GRAPH_SUCCESS;
}

void FillSmallAxisBatched(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, const SmallAxisRoutePlan& plan)
{
    sortTileInfo.ubSize = sortTileInfo.ubSize - SIMT_UB; // reserve 32KB for SIMT kernel scratch
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.lastAxis);
    sortTileInfo.keyParams0 = plan.batchSize;                       // rows per batch
    sortTileInfo.keyParams1 = plan.batchNum;                        // total batches
    sortTileInfo.keyParams2 = plan.useRankInverse ? 1U : 0U;        // enable rank-inverse second pass
    sortTileInfo.keyParams3 = sortTileInfo.isNonLastAxis ? 1U : 0U; // non-last axis flag
    if (sortTileInfo.isNonLastAxis) {
        // Recompute tile count for non-last axis: outerSize * batchSize tiles across all cores.
        uint64_t tileCount64 = 0;
        TryGetSortNonLastTileCount(sortTileInfo, plan.batchSize, sortTileInfo.innerLoopNum, tileCount64);
    }
    sortTileInfo.coreNumNeed = plan.blockDim;
    sortTileInfo.tmpUbSize = plan.tmpUbSize;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
}

// Compute per-row UB footprint for merge-sort multi-core kernel, then derive
// how many rows fit in available UB via ComputeMergeMoreCoreTiling.
ge::graphStatus SetMergeMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    uint32_t byteNum = MERGE_SORT_LIST_NUM * MERGE_SORT_DATA_BYTES * 2;       // value double buffer
    byteNum += MERGE_SORT_LIST_NUM * static_cast<uint32_t>(sizeof(uint32_t)); // int32 index
    if (info.y2DtypeSize == sizeof(int64_t)) {
        byteNum += MERGE_SORT_LIST_NUM * static_cast<uint32_t>(sizeof(int64_t)); // int64 index extra
    }
    byteNum += MERGE_SORT_LIST_NUM * info.dtypeSize;
    OP_CHECK_IF(!ComputeMergeMoreCoreTiling(context, info, byteNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeMergeMoreCoreTiling", "false",
                                                      "The value of ComputeMergeMoreCoreTiling must be true."),
                return ge::GRAPH_FAILED);
    OP_LOGI("[mergeSort]", "maxDealingNum: %u", info.keyParams0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetRadixOneCoreTiling(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    sortTileInfo.lastDimNeedCore = static_cast<uint32_t>(1);
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.lastAxis);
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    uint64_t sortLoopTimes64 = Ops::Base::CeilDiv(sortTileInfo.unsortedDim,
                                                  static_cast<int64_t>(sortTileInfo.maxCoreNum));
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sortLoopTimes",
                                              std::to_string(sortLoopTimes64).c_str(),
                                              "The value of sortLoopTimes must be less than or equal to uint32 max.");
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    if (sortTileInfo.sortLoopTimes > static_cast<uint32_t>(1)) {
        sortTileInfo.coreNumNeed = sortTileInfo.maxCoreNum;
    } else {
        uint32_t core = static_cast<uint32_t>(sortTileInfo.unsortedDim) % sortTileInfo.maxCoreNum;
        sortTileInfo.coreNumNeed = core == uint32_t(0) ? sortTileInfo.maxCoreNum : core;
    }
    sortTileInfo.unsortedDimParallel = sortTileInfo.coreNumNeed;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetRadixMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    OP_CHECK_IF(!FillRadixMoreCoreInfo(info),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "FillRadixMoreCoreInfo", "false",
                                                      "The value of FillRadixMoreCoreInfo must be true."),
                return ge::GRAPH_FAILED);
    info.ubSize = info.ubSize - SIMT_UB;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = info.workspaceSize;
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Fill and print functions
// =============================================================================
void FillTilingDataSort(SortKthTileInfo& info, SortRegBaseTilingData* sortTilingData)
{
    PlanToTilingData(info, sortTilingData);
    sortTilingData->outputIndexRowBytes = info.outputIndexRowBytes;
    return;
}

void PrintTilingDataSort(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    OP_LOGI(context->GetNodeName(),
            "realCoreNum %u, numTileDataSize %u, unsortedDimParallel %u, "
            "lastDimTileNum %u, sortLoopTimes %u, lastDimNeedCore %u, keyParams0 %u, keyParams1 %u "
            "keyParams2 %u, keyParams3 %u, keyParams4 %u, keyParams5 %u, tmpUbSize %u, "
            "lastAxisNum %ld, unsortedDimNum %ld, outerSize %ld, innerSize %ld, innerChunk %u ",
            sortTileInfo.coreNumNeed, sortTileInfo.numTileDataSize, sortTileInfo.unsortedDimParallel,
            sortTileInfo.lastDimTileNum, sortTileInfo.sortLoopTimes, sortTileInfo.lastDimNeedCore,
            sortTileInfo.keyParams0, sortTileInfo.keyParams1, sortTileInfo.keyParams2, sortTileInfo.keyParams3,
            sortTileInfo.keyParams4, sortTileInfo.keyParams5, sortTileInfo.tmpUbSize, sortTileInfo.lastAxis,
            sortTileInfo.unsortedDim, sortTileInfo.outerSize, sortTileInfo.innerSize, sortTileInfo.innerChunk);
    return;
}

ge::graphStatus SetMergeSortTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    OP_CHECK_IF(!ComputeMergeSortTiling(context, info, info.y2DtypeSize),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeMergeSortTiling", "false",
                                                      "The value of ComputeMergeSortTiling must be true."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetMergeIntraCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    OP_CHECK_IF(!ComputeMergeIntraCoreTiling(context, info),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeMergeIntraCoreTiling", "false",
                                                      "The value of ComputeMergeIntraCoreTiling must be true."),
                return ge::GRAPH_FAILED);
    OP_LOGI("MergeIntraCoreTiling",
            "B %ld, N %ld, batchPerCore %u, actualCoreNum %u, blockSortSize %u, extractChunkSize %u, "
            "blocksPerRow %u, alignNum %u, ubSize %u",
            info.unsortedDim, info.lastAxis, info.keyParams0, info.coreNumNeed, info.numTileDataSize, info.keyParams4,
            info.lastDimTileNum, info.keyParams3, info.ubSize);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Try functions
// =============================================================================
bool TrySmallAxis(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (sortTileInfo.lastAxis > static_cast<int64_t>(SMALL_AXIS_THRESHOLD)) {
        return false;
    }
    if (IsAxisOneCopy(sortTileInfo)) {
        if (SetAxisOneCopyTiling(context, sortTileInfo) != ge::GRAPH_SUCCESS) {
            return false;
        }
        schId = SORT_SCHID_7;
        return true;
    }
    SmallAxisRoutePlan smallAxisRoutePlan;
    bool selected = sortTileInfo.isNonLastAxis ? SelectNonLastSmallAxisRoute(sortTileInfo, smallAxisRoutePlan) :
                                                 SelectSmallAxisRoute(sortTileInfo, smallAxisRoutePlan);
    if (!selected) {
        return false;
    }
    // Small-axis schedules also support non-last axes by batching adjacent inner positions.
    // Try them before the generic tile-local transpose path because they have lower setup cost.
    if (smallAxisRoutePlan.kind == SmallAxisRouteKind::TWO_STAGE) {
        schId = SORT_SCHID_6;
    } else if (smallAxisRoutePlan.kind == SmallAxisRouteKind::INSERTION) {
        schId = SORT_SCHID_5;
    } else {
        return false;
    }
    FillSmallAxisBatched(context, sortTileInfo, smallAxisRoutePlan);
    return true;
}

static bool TryAlignBytesToUint32(uint64_t bytes, uint32_t alignBytes, uint32_t& alignedBytes)
{
    if (bytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    uint64_t alignedBytes64 = Ops::Base::CeilAlign<uint64_t>(bytes, static_cast<uint64_t>(alignBytes));
    if (alignedBytes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    alignedBytes = static_cast<uint32_t>(alignedBytes64);
    return true;
}

struct SortNonLastUbLayout {
    uint32_t inputRowBytes = 0;       // one row of input along inner axis
    uint32_t valueAxisBytes = 0;      // sort-axis value buffer per inner position
    uint32_t indexAxisBytes = 0;      // sort-axis index buffer per inner position
    uint32_t outputIndexRowBytes = 0; // one row of output index along inner axis
};

static bool ComputeSortNonLastUbLayout(const SortKthTileInfo& sortTileInfo, uint32_t innerChunk, uint32_t sortCount,
                                       bool useMergeSort, SortNonLastUbLayout& layout)
{
    uint32_t align = sortTileInfo.blockUbSize;
    uint32_t sortDtypeSize = GetNonLastSortDtypeSize(sortTileInfo.dtypeSize, useMergeSort, sortTileInfo.dataType);
    uint64_t valueAxisRawBytes = static_cast<uint64_t>(sortCount) * sortDtypeSize;
    if (useMergeSort) {
        // Advanced merge sort stores value/index pairs internally, so each aligned value row
        // must reserve at least sortCount * 8 bytes even for fp16/bf16 inputs.
        valueAxisRawBytes = std::max(valueAxisRawBytes, static_cast<uint64_t>(sortCount) * SORT_STRUCT_BYTES);
    }
    return TryAlignBytesToUint32(static_cast<uint64_t>(innerChunk) * sortTileInfo.dtypeSize, align,
                                 layout.inputRowBytes) &&
           TryAlignBytesToUint32(valueAxisRawBytes, align, layout.valueAxisBytes) &&
           TryAlignBytesToUint32(static_cast<uint64_t>(sortCount) * sizeof(uint32_t), align, layout.indexAxisBytes) &&
           TryAlignBytesToUint32(static_cast<uint64_t>(innerChunk) * sortTileInfo.y2DtypeSize, align,
                                 layout.outputIndexRowBytes);
}

// Estimate peak UB consumption for a non-last-axis small-axis sort candidate.
// innerChunk: number of adjacent inner positions batched into one sort invocation.
// On success, writes peak UB bytes to peakUb and the chosen layout to candidate.
static bool EstimateSortNonLastSmallAxisUb(SortKthTileInfo& sortTileInfo, uint32_t innerChunk, uint64_t& peakUb,
                                           bool useMergeSort, NonLastSmallAxisCandidate& candidate)
{
    if (innerChunk == 0U) {
        return false;
    }
    uint32_t axisLen = static_cast<uint32_t>(sortTileInfo.lastAxis);
    uint32_t sortCount = GetNonLastSortCount(sortTileInfo.dataType, axisLen);
    SortNonLastUbLayout layout;
    if (!ComputeSortNonLastUbLayout(sortTileInfo, innerChunk, sortCount, useMergeSort, layout)) {
        return false;
    }

    // 8/16-bit gather uses uint16_t offsets represented through signed 16-bit RangeType bit patterns.
    uint64_t inputRowElems = static_cast<uint64_t>(layout.inputRowBytes) / sortTileInfo.dtypeSize;
    uint64_t valueAxisElems = static_cast<uint64_t>(layout.valueAxisBytes) /
                              GetNonLastSortDtypeSize(sortTileInfo.dtypeSize, useMergeSort, sortTileInfo.dataType);
    if (sortTileInfo.dtypeSize <= sizeof(uint16_t) &&
        ((static_cast<uint64_t>(axisLen) - 1U) * inputRowElems > std::numeric_limits<uint16_t>::max() ||
         static_cast<uint64_t>(innerChunk - 1U) * valueAxisElems > std::numeric_limits<uint16_t>::max())) {
        // Keep 16-bit offset bit patterns addressable after reinterpret-casting to uint16_t.
        return false;
    }

    // BF16 merge-sort needs an extra bf16 staging buffer (innerChunk rows) before Cast to fp32.
    uint64_t bf16CastBytes = 0;
    if (useMergeSort && sortTileInfo.dataType == ge::DT_BF16) {
        // BF16 merge uses an extra bf16 staging row before casting to fp32. It is a
        // real producer buffer because its aligned padding is consumed by Cast.
        uint32_t inputValueAxisBytes = 0;
        if (!TryAlignBytesToUint32(static_cast<uint64_t>(sortCount) * sortTileInfo.dtypeSize, sortTileInfo.blockUbSize,
                                   inputValueAxisBytes)) {
            return false;
        }
        bf16CastBytes = static_cast<uint64_t>(innerChunk) * inputValueAxisBytes;
    }

    // Peak UB breakdown:
    //   axisLen * inputRowBytes          — input tile (axisLen rows × innerChunk columns)
    //   innerChunk * valueAxisBytes * 2  — sort input (transposed) + sort output (sorted values)
    //   innerChunk * indexAxisBytes      — index axis (one per inner position)
    //   bf16CastBytes                    — BF16 staging rows (0 for non-BF16 or non-merge)
    peakUb = static_cast<uint64_t>(axisLen) * layout.inputRowBytes +
             static_cast<uint64_t>(innerChunk) * layout.valueAxisBytes * 2U +
             static_cast<uint64_t>(innerChunk) * layout.indexAxisBytes + bf16CastBytes;
    if (innerChunk > 1U) {
        // outputIndex buffer only needed when batching multiple inner positions
        peakUb += static_cast<uint64_t>(axisLen) * layout.outputIndexRowBytes;
    }
    peakUb += static_cast<uint64_t>(sortTileInfo.tmpUbSize);

    // Write back layout to both candidate (for comparison) and sortTileInfo (for kernel)
    candidate.inputRowBytes = layout.inputRowBytes;
    candidate.valueAxisBytes = layout.valueAxisBytes;
    candidate.indexAxisBytes = layout.indexAxisBytes;
    candidate.outputIndexRowBytes = layout.outputIndexRowBytes;
    sortTileInfo.inputRowBytes = layout.inputRowBytes;
    sortTileInfo.valueAxisBytes = layout.valueAxisBytes;
    sortTileInfo.indexAxisBytes = layout.indexAxisBytes;
    sortTileInfo.outputIndexRowBytes = layout.outputIndexRowBytes;
    return true;
}

bool ApplyNonLastSmallAxisResult(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId,
                                 bool useMergeSort, const NonLastSmallAxisCandidate& best, uint64_t usableUb)
{
    sortTileInfo.innerChunk = best.innerChunk;
    sortTileInfo.innerLoopNum = best.innerLoopNum;
    sortTileInfo.coreNumNeed = best.activeCore;
    sortTileInfo.inputRowBytes = best.inputRowBytes;
    sortTileInfo.valueAxisBytes = best.valueAxisBytes;
    sortTileInfo.indexAxisBytes = best.indexAxisBytes;
    sortTileInfo.outputIndexRowBytes = best.outputIndexRowBytes;
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.lastAxis);
    sortTileInfo.lastDimTileNum = 1;
    sortTileInfo.lastDimNeedCore = 1;
    sortTileInfo.sortLoopTimes = 1;
    sortTileInfo.ubSize = sortTileInfo.ubSize - SIMT_UB;
    // Schedule 9 and 10 share the same GM/UB layout. Only the per-row sort primitive
    // differs, so use the selected sort type to choose the binary.
    schId = useMergeSort ? SORT_SCHID_9 : SORT_SCHID_10;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    if (userWorkSpaceSize == nullptr) {
        OP_LOGE(context->GetNodeName(), "get workspace size pointer failed");
        return false;
    }
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    OP_LOGI(context->GetNodeName(),
            "non-last small-axis no-transpose selected axis=%ld inner=%ld outer=%ld innerChunk=%u tileCount=%lu "
            "activeCore=%u peakUb=%lu usableUb=%lu",
            sortTileInfo.lastAxis, sortTileInfo.innerSize, sortTileInfo.outerSize, best.innerChunk, best.tileCount,
            best.activeCore, best.peakUb, usableUb);
    return true;
}

bool TryNonLastSmallAxis(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (!sortTileInfo.isNonLastAxis) {
        return false;
    }
    if (sortTileInfo.lastAxis < NON_LAST_SMALL_AXIS_MIN_AXIS_LEN ||
        sortTileInfo.lastAxis > NON_LAST_SMALL_AXIS_THRESHOLD) {
        return false;
    }
    if (sortTileInfo.innerSize <= 0 || sortTileInfo.outerSize <= 0) {
        return false;
    }
    uint64_t usableUb = ComputeUbAfterSimtReserve(sortTileInfo.ubSize);
    if (usableUb == 0) {
        return false;
    }

    uint32_t sortCount = GetNonLastSortCount(sortTileInfo.dataType, static_cast<uint32_t>(sortTileInfo.lastAxis));
    bool useMergeSort = UseNonLastMergeSort(sortTileInfo.dataType, static_cast<uint32_t>(sortTileInfo.lastAxis));
    // Query Sort tmp with the same aligned sortCount and effective dtype that the
    // kernel will use; BF16 merge sorts as fp32 after the UB cast.
    uint32_t tmpUb = 0;
    GetNonLastSortTmpSize(sortTileInfo.dataType, sortCount, useMergeSort, sortTileInfo.isDescend, tmpUb);
    sortTileInfo.tmpUbSize = tmpUb;

    NonLastSmallAxisCandidate best;
    SortKthTileInfo selectedInfo = sortTileInfo;
    // Bind useMergeSort into a callback so SearchNonLastSmallAxisPlan can evaluate
    // each innerChunk candidate without knowing the sort-type decision.
    auto estimateUb = [useMergeSort](SortKthTileInfo& candidateInfo, uint32_t innerChunk, uint64_t& peakUb,
                                     NonLastSmallAxisCandidate& candidate) -> bool {
        return EstimateSortNonLastSmallAxisUb(candidateInfo, innerChunk, peakUb, useMergeSort, candidate);
    };
    if (!SearchNonLastSmallAxisPlan(sortTileInfo, usableUb, estimateUb, best, &selectedInfo)) {
        OP_LOGI(context->GetNodeName(), "non-last small-axis no-transpose no valid innerChunk");
        return false;
    }
    sortTileInfo = selectedInfo;
    return ApplyNonLastSmallAxisResult(context, sortTileInfo, schId, useMergeSort, best, usableUb);
}

bool TryMerge(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (IsMergeSortSupported(sortTileInfo.dataType, sortTileInfo.lastAxis)) {
        SortKthTileInfo candidate = sortTileInfo;
        if (SetMergeSortTiling(context, candidate) == ge::GRAPH_SUCCESS) {
            sortTileInfo = candidate;
            schId = (sortTileInfo.lastAxis <= SORT32_SMALL_AXIS_THRESHOLD) ? SORT_SCHID_8 : static_cast<uint64_t>(0);
            return true;
        }
    }

    if (IsMergeMoreCoreSupported(sortTileInfo.dataType, sortTileInfo.lastAxis, sortTileInfo.unsortedDim,
                                 sortTileInfo.maxCoreNum)) {
        SortKthTileInfo candidate = sortTileInfo;
        if (SetMergeMoreCoreTiling(context, candidate) == ge::GRAPH_SUCCESS) {
            sortTileInfo = candidate;
            schId = SORT_SCHID_3;
            return true;
        }
    }

    return false;
}

bool TryRadixOneCore(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    SortKthTileInfo candidate = sortTileInfo;
    if (!IsRadixSortOneCore(candidate) || SetRadixOneCoreTiling(context, candidate) != ge::GRAPH_SUCCESS) {
        return false;
    }
    sortTileInfo = candidate;
    schId = static_cast<uint64_t>(1);
    return true;
}

bool TryMergeIntraCore(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (!IsMergeIntraCoreSupported(sortTileInfo.dataType, sortTileInfo.lastAxis, sortTileInfo.unsortedDim,
                                   sortTileInfo.maxCoreNum, sortTileInfo.ubSize)) {
        return false;
    }
    SortKthTileInfo candidate = sortTileInfo;
    if (SetMergeIntraCoreTiling(context, candidate) != ge::GRAPH_SUCCESS) {
        return false;
    }
    sortTileInfo = candidate;
    schId = SORT_SCHID_4;
    return true;
}

// =============================================================================
// Route selection
// =============================================================================
ge::graphStatus SelectSortSchedule(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (TrySmallAxis(context, sortTileInfo, schId)) {
        return ge::GRAPH_SUCCESS;
    }
    if (TryNonLastSmallAxis(context, sortTileInfo, schId)) {
        return ge::GRAPH_SUCCESS;
    }
    if (sortTileInfo.isNonLastAxis) {
        // L0 Sort cannot fall back to full-tensor transpose here. aclnnSort should only
        // dispatch non-last axes that satisfy one of the no-transpose schedules above.
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "sortAxis", std::to_string(sortTileInfo.sortAxis).c_str(),
            "The value of sortAxis must be the last axis or meet no-transpose schedule constraints.");
        return ge::GRAPH_FAILED;
    }
    if (TryMerge(context, sortTileInfo, schId) || TryRadixOneCore(context, sortTileInfo, schId) ||
        TryMergeIntraCore(context, sortTileInfo, schId)) {
        return ge::GRAPH_SUCCESS;
    }

    schId = SORT_SCHID_2;
    OP_CHECK_IF(SetRadixMoreCoreTiling(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SetRadixMoreCoreTiling", "GRAPH_FAILED",
                                                      "The value of SetRadixMoreCoreTiling must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Main entry
// =============================================================================
ge::graphStatus RadixSortTiling(gert::TilingContext* context, int32_t maxCoreNum)
{
    SortRegBaseTilingData* sortTilingData{nullptr};
    sortTilingData = context->GetTilingData<SortRegBaseTilingData>();
    OP_CHECK_IF(sortTilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(sortTilingData, sizeof(SortRegBaseTilingData), 0, sizeof(SortRegBaseTilingData)) != EOK),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "memset_s", "not EOK",
                                                      "The value of memset_s must be EOK."),
                return ge::GRAPH_FAILED);
    SortKthTileInfo sortTileInfo;
    OP_CHECK_IF(SortCheckParams(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SortCheckParams", "GRAPH_FAILED",
                                                      "The value of SortCheckParams must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    sortTileInfo.maxCoreNum = static_cast<uint32_t>(maxCoreNum);
    uint64_t isInt32 = static_cast<uint64_t>(IsRadixUint32CounterRange(sortTileInfo.lastAxis));
    const bool* isDescending = context->GetAttrs()->GetAttrPointer<bool>(1);
    uint64_t isDescend = *isDescending;
    sortTileInfo.isDescend = static_cast<bool>(isDescend);
    sortTileInfo.isInt32 = static_cast<uint32_t>(isInt32);
    OP_LOGI(context->GetNodeName(), "isInt32 is %lu, isDescend is %lu", isInt32, isDescend);
    uint64_t schId = static_cast<uint64_t>(0);
    OP_CHECK_IF(SelectSortSchedule(context, sortTileInfo, schId) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SelectSortSchedule", "GRAPH_FAILED",
                                                      "The value of SelectSortSchedule must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, isInt32, isDescend);
    OP_LOGI(context->GetNodeName(), "tilingKey is %lu, maxCoreNum %d, schId %lu", tilingKey, maxCoreNum, schId);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(sortTileInfo.coreNumNeed);
    context->SetLocalMemorySize(sortTileInfo.ubSize);
    FillTilingDataSort(sortTileInfo, sortTilingData);
    PrintTilingDataSort(context, sortTileInfo);
    OP_LOGI(context->GetNodeName(), "end RadixSortTIling ");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortTilingSimt(gert::TilingContext* context, int32_t maxCoreNum)
{
    return RadixSortTiling(context, maxCoreNum);
}
} // namespace optiling
