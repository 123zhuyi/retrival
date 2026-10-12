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
 * \file top_k_v2_tiling_base.cpp
 * \brief top_k_v2 common tiling helpers implementation
 */
#include "top_k_v2_tiling_base.h"

#include <algorithm>
#include <string>

#include "log/log.h"
#include "util/platform_util.h"

namespace optiling {
namespace topkV2 {

// ==================== Helper Functions ====================

uint32_t GetDataTypeSize(ge::DataType dataType) { return topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second; }

bool IsDataType64Bit(ge::DataType dataType) { return topkV2DataInfo::b64DataTypeBitMap.count(dataType) != 0; }

uint32_t GetDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::TMP_DATA_NUM_B64 : topkV2DataInfo::TMP_DATA_NUM;
}

uint32_t GetSingleBlockModelDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::SINGLE_BLOCK_DATA_NUM_B64 :
                                       topkV2DataInfo::SINGLE_BLOCK_DATA_NUM;
}

uint32_t GetSingleCoreModelDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::SINGLE_CORE_DATA_NUM_B64 : topkV2DataInfo::SINGLE_CORE_DATA_NUM;
}

// ==================== Common Align Helpers ====================

bool CeilAlignUint32(uint64_t rawSize, uint32_t alignSize, uint32_t& alignedSize)
{
    uint64_t result = Ops::Base::CeilAlign(rawSize, static_cast<uint64_t>(alignSize));
    if (result > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    alignedSize = static_cast<uint32_t>(result);
    return true;
}

// ==================== FP32 MergeSort Helpers ====================

uint32_t AlignTopkMergeMoreCoreWorkspaceElems(int64_t elementNum)
{
    if (elementNum <= 0) {
        return 0;
    }
    return static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(elementNum * topkV2DataInfo::SORT_STRUCT_SIZE_FP32),
                             topkV2DataInfo::AGLIN_FACTOR) /
        topkV2DataInfo::SORT_STRUCT_SIZE_FP32);
}

uint32_t ComputeTopkMergeMoreCoreOnceMaxElements(uint64_t ubSizePlatForm, ge::DataType indicesDType)
{
    uint32_t indexBytes = GetDataTypeSize(indicesDType);
    uint32_t bytesPerElem = topkV2DataInfo::MERGE_MORE_CORE_LIST_MAX_NUM * topkV2DataInfo::SORT_STRUCT_SIZE_FP32 *
                            topkV2DataInfo::CONST_TWO;
    bytesPerElem += topkV2DataInfo::MERGE_MORE_CORE_LIST_MAX_NUM * static_cast<uint32_t>(sizeof(uint32_t));
    bytesPerElem += topkV2DataInfo::MERGE_MORE_CORE_LIST_MAX_NUM * static_cast<uint32_t>(sizeof(float));
    if (indexBytes == topkV2DataInfo::INT64_BYTE) {
        bytesPerElem += topkV2DataInfo::MERGE_MORE_CORE_LIST_MAX_NUM * indexBytes;
    }
    return bytesPerElem == 0 ? 0 : static_cast<uint32_t>(ubSizePlatForm / bytesPerElem);
}

uint32_t ComputeTopkMergeIntraCoreBlockSortSize(uint64_t ubSizePlatForm)
{
    constexpr uint32_t phase2BytesPerElem = topkV2DataInfo::CONST_TWO * topkV2DataInfo::SORT_STRUCT_SIZE_FP32 *
                                            topkV2DataInfo::CONST_TWO * topkV2DataInfo::CONST_TWO;
    uint32_t blockSortSize = static_cast<uint32_t>(ubSizePlatForm / phase2BytesPerElem);
    return (blockSortSize / topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN) * topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN;
}

uint32_t ComputeTopkMergeIntraCoreExtractChunkSize(uint64_t ubSizePlatForm, ge::DataType indicesDType)
{
    uint32_t indexBytes = GetDataTypeSize(indicesDType);
    uint32_t bytesPerElem = (topkV2DataInfo::SORT_STRUCT_SIZE_FP32 + sizeof(float) + sizeof(int32_t) + indexBytes) *
                            topkV2DataInfo::CONST_TWO;
    uint32_t extractChunkSize = static_cast<uint32_t>(ubSizePlatForm / bytesPerElem);
    return (extractChunkSize / topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN) *
           topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN;
}

// ==================== NonLastSmallAxis Helpers ====================

uint32_t GetTopkPreferredInnerChunk(ge::DataType dataType, uint32_t index)
{
    static constexpr uint32_t CHUNK_CANDIDATES[][topkV2DataInfo::MAX_INNER_CHUNK_CANDIDATES] = {
        {4, 2, 1, 0, 0, 0},
        {8, 4, 2, 1, 0, 0},
        {16, 8, 4, 2, 1, 0},
        {32, 16, 8, 4, 2, 1},
    };
    static constexpr uint32_t CHUNK_VALID_COUNT[] = {3, 4, 5, 6};
    uint32_t group = 0;
    if (dataType == ge::DT_INT64 || dataType == ge::DT_UINT64) {
        group = topkV2DataInfo::INNER_CHUNK_GROUP_8BYTE;
    } else if (dataType == ge::DT_FLOAT || dataType == ge::DT_INT32 || dataType == ge::DT_UINT32) {
        group = topkV2DataInfo::INNER_CHUNK_GROUP_4BYTE;
    } else if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16 || dataType == ge::DT_INT16 ||
               dataType == ge::DT_UINT16) {
        group = topkV2DataInfo::INNER_CHUNK_GROUP_2BYTE;
    } else if (dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
        group = topkV2DataInfo::INNER_CHUNK_GROUP_1BYTE;
    } else {
        return 0;
    }
    return index < CHUNK_VALID_COUNT[group] ? CHUNK_CANDIDATES[group][index] : 0;
}

bool UseTopkNonLastMergeSort(ge::DataType dataType, uint32_t axisLen)
{
    return dataType == ge::DT_FLOAT ||
           ((dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) && axisLen <= topkV2DataInfo::SMALL_MAX_DATA_SZIE);
}

ge::DataType GetTopkNonLastSortDtype(ge::DataType dataType, bool useMergeSort)
{
    return useMergeSort && dataType == ge::DT_BF16 ? ge::DT_FLOAT : dataType;
}

uint32_t GetTopkNonLastSortDtypeSize(uint32_t dtypeSize, bool useMergeSort, ge::DataType dataType)
{
    return useMergeSort && dataType == ge::DT_BF16 ? static_cast<uint32_t>(sizeof(float)) : dtypeSize;
}

bool GetTopkNonLastSortTmpSize(ge::DataType dataType, uint32_t sortCount, bool useMergeSort, bool isDescend,
                               uint32_t& tmpUbSize)
{
    std::vector<int64_t> shapeVec = {static_cast<int64_t>(sortCount)};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = useMergeSort ? AscendC::SortType::MERGE_SORT : AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    ge::DataType finalDataType = GetTopkNonLastSortDtype(dataType, useMergeSort);
    AscendC::GetSortMaxMinTmpSize(srcShape, finalDataType, ge::DT_UINT32, true, config, maxValue, minValue);
    tmpUbSize = maxValue;
    return maxValue > 0;
}

void ComputeTopkAxisDimProducts(const gert::Shape& shape, int64_t axis, TopkNonLastSmallAxisTileInfo& info)
{
    int64_t rank = shape.GetDimNum();
    if (axis < 0 || axis >= rank) {
        return;
    }
    int64_t outerSize = 1;
    int64_t innerSize = 1;
    for (int64_t i = 0; i < rank; ++i) {
        int64_t dimSize = shape.GetDim(i);
        if (i < axis) {
            outerSize *= dimSize;
        } else if (i > axis) {
            innerSize *= dimSize;
        }
    }
    info.outerSize = outerSize;
    info.innerSize = innerSize;
    info.lastAxis = shape.GetDim(axis);
    info.unsortedDim = outerSize * innerSize;
}

// ==================== TopK API Buffer Calculation ====================

ge::graphStatus GetTopkApiTmpBufferSize(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                        uint32_t needDataNum, int64_t kValue, bool isLargest, ge::DataType dtype,
                                        bool isSort, uint32_t nowTileSize)
{
    int32_t aglinInnerValue = static_cast<int32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(needDataNum), topkV2DataInfo::AGLIN_FACTOR));

    uint32_t aglinKValue = (topkTilingData.get_modeType() == topkV2DataInfo::SINGLE_CORE_MODE) ?
                               std::min(static_cast<int64_t>(needDataNum), kValue) :
                               std::min(static_cast<int64_t>(nowTileSize), kValue);

    AscendC::TopKConfig topkConfig;
    topkConfig.algo = AscendC::TopKAlgo::RADIX_SELECT;
    topkConfig.order = AscendC::TopKOrder::UNSET;
    topkConfig.sorted = isSort;

    uint32_t maxBufferSize = 0;
    uint32_t minBufferSize = 0;
    bool isSuccess = AscendC::GetTopKMaxMinTmpSize(aglinInnerValue, 1, aglinKValue, false, false,
                                                   AscendC::TopKMode::TOPK_NORMAL, isLargest, dtype, topkConfig,
                                                   maxBufferSize, minBufferSize);

    OP_LOGI("TopKV2TilingForAscendC", "TopK API buffer: kValue=%ld, alignedK=%u, alignedInner=%u, bufferSize=%u",
            kValue, aglinKValue, aglinInnerValue, maxBufferSize);

    OP_CHECK_IF(!isSuccess,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "GetTopKMaxMinTmpSize", "false",
                                                      "The value of GetTopKMaxMinTmpSize must be true."),
                return ge::GRAPH_FAILED);

    topkTilingData.set_topkAcApiTmpBufferSize(maxBufferSize);
    return ge::GRAPH_SUCCESS;
}

// ==================== Runtime Space Calculation ====================

uint64_t GetTopkMultiCoreRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t maxCoreNum,
                                          uint32_t xDtypeSize, uint32_t indexToDtypeSize, uint32_t indexDtypeSize,
                                          int64_t kValue)
{
    OP_CHECK_IF(tileData == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "tileData", std::to_string(tileData).c_str(),
                                                      "The value of tileData must be greater than 0."),
                return ge::GRAPH_FAILED);

    uint64_t aglinFactor = topkV2DataInfo::AGLIN_FACTOR;
    uint32_t lastDimTileNum = (static_cast<uint32_t>(lastAxisNum) + tileData - 1) / tileData;
    uint32_t lastDimTileNumTimes = (lastDimTileNum + maxCoreNum - 1) / maxCoreNum;
    uint64_t lastDimTileNumTimesAlign = Ops::Base::CeilAlign(
        static_cast<uint64_t>(sizeof(uint32_t) * lastDimTileNumTimes), aglinFactor);
    uint64_t initUb = indexDtypeSize * topkV2DataInfo::BIN_NUM * (lastDimTileNumTimes + 1) +
                      lastDimTileNumTimesAlign * topkV2DataInfo::CONST_TWO;

    uint32_t factor = xDtypeSize * topkV2DataInfo::CONST_TWO + indexDtypeSize + indexToDtypeSize;

    if (tileData < kValue) {
        factor += xDtypeSize + indexToDtypeSize + sizeof(int32_t);
    } else {
        initUb += Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)), aglinFactor) +
                  Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize), aglinFactor) +
                  Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize), aglinFactor);
    }
    OP_LOGI("TopKV2TilingForAscendC", "tileData=%u, initUb=%u, factor = %u", tileData, initUb, factor);
    return initUb + factor * tileData;
}

uint64_t GetSingleBlockTopkRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize,
                                            uint32_t indexToDtypeSize, int64_t kValue)
{
    OP_CHECK_IF(lastAxisNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "lastAxisNum", std::to_string(lastAxisNum).c_str(),
                                                      "The value of lastAxisNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    uint32_t batchNumInUb = tileData / lastAxisNum;
    uint64_t alignTileData = Ops::Base::CeilAlign(static_cast<uint64_t>(lastAxisNum), topkV2DataInfo::AGLIN_FACTOR);
    uint64_t alignkValueMultDtypeSize = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize),
                                                             topkV2DataInfo::AGLIN_FACTOR);
    uint64_t alignkValueMultIndexDtypeSize = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize),
                                                                  topkV2DataInfo::AGLIN_FACTOR);
    uint64_t alignIndicesOutTbuf = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)),
                                                        topkV2DataInfo::AGLIN_FACTOR);
    uint64_t initUb = batchNumInUb * (alignTileData * xDtypeSize + alignkValueMultDtypeSize +
                                      alignkValueMultIndexDtypeSize + alignIndicesOutTbuf);
    OP_LOGD("TopKV2TilingForAscendC",
            "compute single block alignTileData=%u, alignkValueMultDtypeSize=%u, "
            "alignkValueMultIndexDtypeSize=%u, alignIndicesOutTbuf=%u.",
            alignTileData, alignkValueMultDtypeSize, alignkValueMultIndexDtypeSize, alignIndicesOutTbuf);
    return initUb;
}

uint64_t GetTopkMultiCoreOptimModeRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize,
                                                   uint32_t indexToDtypeSize, int64_t kValue, uint64_t ubBlockAlignSize)
{
    uint64_t dataSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(tileData), ubBlockAlignSize) * xDtypeSize;
    uint64_t indexSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(tileData), ubBlockAlignSize) * sizeof(int32_t);
    uint64_t topkOutDataSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * xDtypeSize), ubBlockAlignSize);
    uint64_t topkOutIndexSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * sizeof(int32_t)),
                                                      ubBlockAlignSize);
    uint64_t tempConversionSpace = Ops::Base::CeilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize),
                                                        ubBlockAlignSize);
    uint64_t initUb = dataSpace + indexSpace + topkOutDataSpace + topkOutIndexSpace + tempConversionSpace;
    OP_LOGI(
        "TopKV2TilingForAscendC",
        "compute runTime space lastAxisNum =%u, tileData=%lu, xDtypeSize=%u, indexToDtypeSize=%u, kValue=%u, initUb=%u",
        lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue, initUb);
    return initUb;
}

uint64_t GetSingleCoreTopkRunTimeNeedSpace(int64_t lastAxisNum, uint32_t nowTileSize, uint32_t xDtypeSize,
                                           uint32_t indexToDtypeSize, int64_t kValue, bool isSort)
{
    OP_CHECK_IF(nowTileSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "nowTileSize", std::to_string(nowTileSize).c_str(),
                                                      "The value of nowTileSize must be greater than 0."),
                return ge::GRAPH_FAILED);

    auto ceilAlign = [](uint64_t value) -> uint64_t {
        return Ops::Base::CeilAlign(value, topkV2DataInfo::AGLIN_FACTOR);
    };

    int64_t lastDimTileNum = (lastAxisNum + nowTileSize - 1) / nowTileSize;
    uint32_t tileNum = lastAxisNum / lastDimTileNum;
    uint32_t tailTileNum = lastAxisNum % lastDimTileNum;
    tileNum = tailTileNum == 0 ? tileNum : tileNum + 1;
    uint32_t outQueueNum = std::min(tileNum, static_cast<uint32_t>(kValue));

    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    uint32_t indexTypeSize = (lastAxisNum <= int32Max) ? sizeof(int32_t) : sizeof(int64_t);

    uint64_t initUb = 0;
    initUb += ceilAlign(tileNum) * xDtypeSize;
    initUb += ceilAlign(outQueueNum * xDtypeSize);
    initUb += ceilAlign(outQueueNum * indexToDtypeSize);
    initUb += ceilAlign(outQueueNum * sizeof(int32_t));
    initUb += topkV2DataInfo::BIN_NUM * sizeof(int32_t);
    initUb += topkV2DataInfo::BIN_NUM * indexTypeSize;
    initUb += ceilAlign(static_cast<uint64_t>(lastDimTileNum) * sizeof(int32_t));
    initUb += ceilAlign(tileNum * sizeof(int32_t));
    initUb += topkV2DataInfo::BIN_NUM * indexTypeSize;

    if (isSort && kValue * xDtypeSize <= topkV2DataInfo::SUPPORT_SORT_MAX_BYTE_SIZE) {
        initUb += ceilAlign(static_cast<uint64_t>(kValue * indexToDtypeSize));
    }

    return initUb;
}

// ==================== MergeSort Helpers ====================

bool IsLastLoopCoreUtilizationSuccess(uint64_t unsortedDimNum, uint32_t tmpOneCoreRowNum, uint32_t maxCoreNum)
{
    uint64_t virUnsortedDimNeedCoreNum = (unsortedDimNum + tmpOneCoreRowNum - 1) / tmpOneCoreRowNum;
    uint64_t sortLoopTimes = (virUnsortedDimNeedCoreNum + maxCoreNum - 1) / maxCoreNum;
    uint32_t lastLoopDimNum = static_cast<uint32_t>(unsortedDimNum %
                                                    (static_cast<uint64_t>(maxCoreNum) * tmpOneCoreRowNum));
    uint32_t lastLoopDimNeedCoreNum = lastLoopDimNum / tmpOneCoreRowNum;
    if (lastLoopDimNum == 0) {
        return true;
    }
    bool loopTimesCondition = sortLoopTimes >= topkV2DataInfo::SMALL_LOOP_LOWER_NUM &&
                              sortLoopTimes <= topkV2DataInfo::SMALL_LOOP_UPPER_NUM;
    bool utilizationCondition = lastLoopDimNeedCoreNum < maxCoreNum * topkV2DataInfo::LAST_LOOP_CORE_UTILIZATION;
    if (loopTimesCondition && utilizationCondition) {
        return false;
    }
    return true;
}

uint32_t GetTileDataForMergeSort(uint64_t unsortedDimNum, uint32_t maxCoreNum, uint32_t tileMaxData, uint32_t bufferNum,
                                 uint32_t aglinNum)
{
    OP_CHECK_IF(bufferNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "bufferNum", std::to_string(bufferNum).c_str(),
                                                      "The value of bufferNum must be greater than 0."),
                return topkV2DataInfo::SMALL_MAX_DATA_SZIE);
    OP_CHECK_IF(aglinNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "aglinNum", std::to_string(aglinNum).c_str(),
                                                      "The value of aglinNum must be greater than 0."),
                return topkV2DataInfo::SMALL_MAX_DATA_SZIE);

    uint32_t tileData = topkV2DataInfo::TMP_DATA_NUM;
    uint32_t oneCoreRowNum = (tileData / bufferNum) / aglinNum;
    oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
    uint64_t virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;

    if (virUnsortedDimNeedCoreNum < maxCoreNum) {
        oneCoreRowNum = (unsortedDimNum + maxCoreNum - 1) / maxCoreNum;
        oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
        virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
        tileData = oneCoreRowNum * bufferNum * aglinNum;
        tileData = std::min(tileData, tileMaxData - topkV2DataInfo::BIN_NUM);
        return tileData;
    }

    while (virUnsortedDimNeedCoreNum >= maxCoreNum && topkV2DataInfo::BIN_NUM + tileData < tileMaxData) {
        tileData += topkV2DataInfo::BIN_NUM;
        oneCoreRowNum = (tileData / bufferNum) / aglinNum;
        oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
        virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
    }

    uint32_t tmpTileData = tileData;
    while (!IsLastLoopCoreUtilizationSuccess(unsortedDimNum, oneCoreRowNum, maxCoreNum)) {
        if (tileData < topkV2DataInfo::BIN_NUM) {
            OP_LOGD("TopKV2TilingForAscendC", "tileData optimization =%u", tmpTileData);
            return tmpTileData;
        }
        tileData -= topkV2DataInfo::BIN_NUM;
        oneCoreRowNum = (tileData / bufferNum) / aglinNum;
        oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
        virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
    }

    return tileData;
}

void SetMergeSortTmpSize(gert::TilingContext* context, ge::DataType dataType, int64_t lastAxisNum,
                         TopKV2TilingDataSimd& topkTilingData)
{
    auto platform_info = context->GetPlatformInfo();
    if (nullptr == platform_info) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platform_info");
    }

    uint32_t alignDataSize = (static_cast<uint32_t>(lastAxisNum) + topkV2DataInfo::AGLIN_FACTOR - 1) /
                             topkV2DataInfo::AGLIN_FACTOR * topkV2DataInfo::AGLIN_FACTOR;
    uint32_t dataTypeSize = (dataType == ge::DT_BF16) ? GetDataTypeSize(ge::DT_FLOAT) : GetDataTypeSize(dataType);

    auto plat = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t dataSizeNeed = AscendC::GetConcatTmpSize(plat, alignDataSize, dataTypeSize);
    OP_LOGI("TopKV2TilingForAscendC", "Allocal buffer mergesort element len = %ld ac merge api", lastAxisNum);
    OP_LOGI("TopKV2TilingForAscendC", "Merge sort need tmp buffer %u byte for ac merge api", dataSizeNeed);
    topkTilingData.set_mergSortAcApiNeedBufferSize(dataSizeNeed);
}

// ==================== Mode Judgment Functions ====================

bool needSortWithIndex(TopKV2TilingDataSimd& topkTilingData, bool isSorted, ge::DataType dataType)
{
    if (isSorted && topkTilingData.get_modeType() == topkV2DataInfo::MULT_CORE_MODE) {
        if (topkTilingData.get_topKRealValue() <= topkV2DataInfo::SUPPORT_SORT_MAX_SIZE) {
            return false;
        }
        return true;
    }
    uint32_t xDtypeSize = static_cast<uint32_t>(topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second);
    if (isSorted && topkTilingData.get_modeType() == topkV2DataInfo::SINGLE_CORE_MODE) {
        if (topkTilingData.get_topKRealValue() <= topkV2DataInfo::SUPPORT_SORT_MAX_SIZE &&
            topkTilingData.get_topKRealValue() * xDtypeSize <= topkV2DataInfo::SUPPORT_SORT_MAX_BYTE_SIZE) {
            return false;
        }
        return true;
    }
    return false;
}

bool IsBitonicSmallTopkMode(int64_t kValue, int64_t sortPolicy, bool isSort)
{
    return isSort && static_cast<uint64_t>(kValue) >= topkV2DataInfo::BITONIC_SMALL_TOPK_MIN_K &&
           static_cast<uint64_t>(kValue) <= topkV2DataInfo::BITONIC_SMALL_TOPK_MAX_K &&
           static_cast<uint64_t>(sortPolicy) == topkV2DataInfo::BITONIC_SMALL_TOPK_POLICY;
}

// ==================== NonLastSmallAxis Calculation Helpers ====================

bool SearchTopkNonLastSmallAxisPlan(
    const TopkNonLastSmallAxisTileInfo& info, uint64_t usableUb,
    std::function<bool(TopkNonLastSmallAxisTileInfo&, uint32_t, uint64_t&, TopkNonLastSmallAxisCandidate&)> estimateUb,
    TopkNonLastSmallAxisCandidate& best, TopkNonLastSmallAxisTileInfo* selectedInfo)
{
    for (uint32_t i = 0; i < topkV2DataInfo::MAX_INNER_CHUNK_CANDIDATES; ++i) {
        uint32_t chunk = GetTopkPreferredInnerChunk(info.dataType, i);
        if (chunk == 0U) {
            break;
        }
        chunk = static_cast<uint32_t>(std::min<uint64_t>(chunk, static_cast<uint64_t>(info.innerSize)));
        if (chunk == 0U) {
            return false;
        }
        uint64_t innerLoopNum64 = (static_cast<uint64_t>(info.innerSize) + chunk - 1U) / chunk;
        if (innerLoopNum64 == 0U || innerLoopNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }
        TopkNonLastSmallAxisCandidate cur;
        cur.innerChunk = chunk;
        cur.innerLoopNum = static_cast<uint32_t>(innerLoopNum64);
        cur.tileCount = static_cast<uint64_t>(info.outerSize) * innerLoopNum64;
        if (cur.tileCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }
        TopkNonLastSmallAxisTileInfo candidateInfo = info;
        if (!estimateUb(candidateInfo, chunk, cur.peakUb, cur) || cur.peakUb > usableUb) {
            continue;
        }
        cur.activeCore = static_cast<uint32_t>(
            std::min<uint64_t>(static_cast<uint64_t>(info.maxCoreNum), cur.tileCount));
        bool betterCoreUse = cur.activeCore > best.activeCore;
        bool sameCoreUseLargerChunk = cur.activeCore == best.activeCore && cur.innerChunk > best.innerChunk;
        if (betterCoreUse || sameCoreUseLargerChunk) {
            best = cur;
            if (selectedInfo != nullptr) {
                *selectedInfo = candidateInfo;
            }
        }
    }
    return best.innerChunk != 0U && best.tileCount != 0U && best.activeCore != 0U;
}

bool ComputeTopkNonLastLayout(const TopkNonLastSmallAxisTileInfo& info, uint32_t kValue, uint32_t innerChunk,
                              bool useMergeSort, topkV2DataInfo::NonLastSmallAxisTopkLayout& layout)
{
    if (innerChunk == 0U || kValue == 0U || info.dtypeSize == 0U || info.blockUbSize == 0U) {
        return false;
    }
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    uint32_t sortCount = Ops::Base::CeilAlign(axisLen, topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN);
    uint32_t sortDtypeSize = GetTopkNonLastSortDtypeSize(info.dtypeSize, useMergeSort, info.dataType);
    uint64_t valueAxisRawBytes = static_cast<uint64_t>(sortCount) * sortDtypeSize;
    uint32_t outputCount = useMergeSort ? sortCount : kValue;
    if (useMergeSort) {
        valueAxisRawBytes = std::max(valueAxisRawBytes,
                                     static_cast<uint64_t>(sortCount) * topkV2DataInfo::SORT_STRUCT_BYTES);
    }

    uint64_t valueOutputRawBytes = static_cast<uint64_t>(outputCount) * sortDtypeSize;
    if (useMergeSort) {
        valueOutputRawBytes = std::max(valueOutputRawBytes,
                                       static_cast<uint64_t>(outputCount) * topkV2DataInfo::SORT_STRUCT_BYTES);
    }
    if (!CeilAlignUint32(static_cast<uint64_t>(innerChunk) * info.dtypeSize, info.blockUbSize, layout.inputRowBytes) ||
        !CeilAlignUint32(valueAxisRawBytes, info.blockUbSize, layout.axisRowBytes) ||
        !CeilAlignUint32(valueOutputRawBytes, info.blockUbSize, layout.valueRowBytes) ||
        !CeilAlignUint32(static_cast<uint64_t>(outputCount) * sizeof(uint32_t), info.blockUbSize,
                         layout.indexRowBytes)) {
        return false;
    }

    uint64_t inputRowElems = static_cast<uint64_t>(layout.inputRowBytes) / info.dtypeSize;
    uint64_t axisRowElems = static_cast<uint64_t>(layout.axisRowBytes) / sortDtypeSize;
    if (info.dtypeSize <= sizeof(uint16_t) &&
        ((static_cast<uint64_t>(axisLen) - 1U) * inputRowElems > std::numeric_limits<uint16_t>::max() ||
         static_cast<uint64_t>(innerChunk - 1U) * axisRowElems > std::numeric_limits<uint16_t>::max())) {
        return false;
    }
    return true;
}

bool EstimateTopkNonLastSmallAxisUb(TopkNonLastSmallAxisTileInfo& info, uint32_t kValue, uint32_t innerChunk,
                                    bool useMergeSort, uint64_t& peakUb, TopkNonLastSmallAxisCandidate& candidate)
{
    topkV2DataInfo::NonLastSmallAxisTopkLayout layout;
    if (!ComputeTopkNonLastLayout(info, kValue, innerChunk, useMergeSort, layout)) {
        return false;
    }
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    uint32_t sortCount = Ops::Base::CeilAlign(axisLen, topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN);

    uint32_t inputCastRowBytes = 0;
    if (useMergeSort && info.dataType == ge::DT_BF16 &&
        !CeilAlignUint32(static_cast<uint64_t>(sortCount) * info.dtypeSize, info.blockUbSize, inputCastRowBytes)) {
        return false;
    }

    peakUb = static_cast<uint64_t>(axisLen) * layout.inputRowBytes +
             static_cast<uint64_t>(innerChunk) * layout.axisRowBytes +
             static_cast<uint64_t>(innerChunk) * layout.valueRowBytes +
             static_cast<uint64_t>(innerChunk) * layout.indexRowBytes +
             static_cast<uint64_t>(innerChunk) * inputCastRowBytes + static_cast<uint64_t>(info.tmpUbSize);
    candidate.inputRowBytes = layout.inputRowBytes;
    candidate.valueAxisBytes = layout.axisRowBytes;
    candidate.indexAxisBytes = layout.valueRowBytes;
    candidate.outputIndexRowBytes = layout.indexRowBytes;
    info.inputRowBytes = layout.inputRowBytes;
    info.valueAxisBytes = layout.axisRowBytes;
    info.indexAxisBytes = layout.valueRowBytes;
    info.outputIndexRowBytes = layout.indexRowBytes;
    return true;
}

// ==================== NonLastSmallAxis Init and Search ====================

bool InitTopkNonLastSmallAxisInfo(gert::TilingContext* context, const gert::Shape& inputShape, int32_t axis,
                                  const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                  TopkNonLastSmallAxisTileInfo& info)
{
    info.rank = inputShape.GetDimNum();
    info.sortAxis = axis;
    info.maxCoreNum = computeInfo.maxCoreNum;
    info.dataType = computeInfo.dataType;
    info.dtypeSize = GetDataTypeSize(computeInfo.dataType);
    info.y2DtypeSize = GetDataTypeSize(computeInfo.indicesDType);
    info.blockUbSize = static_cast<uint32_t>(computeInfo.ubBlockAlignSize);
    ComputeTopkAxisDimProducts(inputShape, axis, info);

    if (info.lastAxis <= 0 || info.innerSize <= 0 || info.outerSize <= 0 ||
        info.lastAxis > topkV2DataInfo::NON_LAST_SMALL_AXIS_THRESHOLD || computeInfo.kValue <= 0 ||
        computeInfo.kValue > info.lastAxis) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "axis, k",
            (std::to_string(info.lastAxis) + ", " + std::to_string(computeInfo.kValue)).c_str(),
            "The value of axis must be positive and less than or equal to threshold, and the value of k must be within "
            "the range [1, axis].");
        return false;
    }
    return true;
}

ge::graphStatus SetupTopkNonLastSmallAxisTmpUb(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                               const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                               TopkNonLastSmallAxisTileInfo& info, bool useMergeSort,
                                               uint32_t sortCount)
{
    if (useMergeSort) {
        uint32_t tmpUbSize = 0;
        if (!GetTopkNonLastSortTmpSize(computeInfo.dataType, sortCount, useMergeSort, computeInfo.isLargest,
                                       tmpUbSize)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "GetTopkNonLastSortTmpSize", "false",
                                                  "The value of GetTopkNonLastSortTmpSize must be true.");
            return ge::GRAPH_FAILED;
        }
        info.tmpUbSize = tmpUbSize;
        topkTilingData.set_tmpUbSize(tmpUbSize);
        topkTilingData.set_topkAcApiTmpBufferSize(0);
    } else {
        if (GetTopkApiTmpBufferSize(context, topkTilingData, static_cast<uint32_t>(info.lastAxis), computeInfo.kValue,
                                    computeInfo.isLargest, computeInfo.dataType, computeInfo.isSort,
                                    static_cast<uint32_t>(info.lastAxis)) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        info.tmpUbSize = topkTilingData.get_topkAcApiTmpBufferSize();
        topkTilingData.set_tmpUbSize(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SearchBestTopkNonLastSmallAxisPlan(gert::TilingContext* context, TopkNonLastSmallAxisTileInfo& info,
                                                   const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                                   bool useMergeSort, TopkNonLastSmallAxisCandidate& best)
{
    TopkNonLastSmallAxisTileInfo selectedInfo = info;
    auto estimateUb = [kValue = static_cast<uint32_t>(computeInfo.kValue), useMergeSort](
                          TopkNonLastSmallAxisTileInfo& candidateInfo, uint32_t innerChunk, uint64_t& peakUb,
                          TopkNonLastSmallAxisCandidate& candidate) -> bool {
        return EstimateTopkNonLastSmallAxisUb(candidateInfo, kValue, innerChunk, useMergeSort, peakUb, candidate);
    };

    if (!SearchTopkNonLastSmallAxisPlan(info, computeInfo.ubSizePlatForm, estimateUb, best, &selectedInfo)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SearchTopkNonLastSmallAxisPlan", "false",
                                              "The value of SearchTopkNonLastSmallAxisPlan must be true.");
        return ge::GRAPH_FAILED;
    }
    info = selectedInfo;
    return ge::GRAPH_SUCCESS;
}

} // namespace topkV2
} // namespace optiling
