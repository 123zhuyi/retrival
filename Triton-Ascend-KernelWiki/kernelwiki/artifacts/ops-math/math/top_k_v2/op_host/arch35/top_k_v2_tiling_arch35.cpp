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
 * \file top_k_v2_tiling_arch35.cpp
 * \brief top_k_v2 impl
 */
#include "top_k_v2_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "sort_with_index_tiling.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace optiling {
namespace topkV2 {
namespace topkV2DataInfo {
const uint32_t CONST_ZERO = 0;
const uint32_t CONST_TWO = 2;
const uint32_t CONST_THREE = 3;
const uint32_t MAX_K_FOR_INT64 = 2000;
const uint32_t BIN_NUM = 256;
const uint32_t TILE_SIZE_DECREASING_FACTOR = 32;
const uint32_t TMP_DATA_NUM = 7680; // 默认UB一次性能处理的非64位数据的个数，可根据场景动态调整
const uint32_t TMP_DATA_NUM_B64 = 5120; // 默认UB一次性能处理的64位数据的个数，可根据场景动态调整
const uint32_t SINGLE_BLOCK_DATA_NUM = 15360;
const uint32_t SINGLE_BLOCK_DATA_NUM_B64 = 10240;
const uint32_t SINGLE_CORE_DATA_NUM = 15360;
const uint32_t SINGLE_CORE_DATA_NUM_B64 = 10240;
const uint64_t AGLIN_FACTOR = 32;
const uint32_t SMALL_MAX_DATA_SZIE = 1024;
const uint32_t MERGE_SORT_TILING_OFFSET = 10000;
const uint32_t SINGLE_CORE_MODE = 1;
const uint32_t MULT_CORE_MODE = 2;
const uint32_t MULT_CORE_OPTIM_MODE = 4;
const uint32_t SINGLE_BLOCK_MODE = 3;
const uint32_t SORT_AND_TOP_K_MODE = 5;
const uint32_t FP32_MERGE_MORE_CORE_MODE = 6;
const uint32_t FP32_MERGE_INTRA_CORE_MODE = 7;
const uint32_t NON_LAST_SMALL_AXIS_MODE = 8;
const uint32_t NON_LAST_SMALL_AXIS_RADIX_SELECT = 0;
const uint32_t NON_LAST_SMALL_AXIS_MERGE_SORT = 1;
const uint32_t INT64_BYTE = 8;
const uint32_t INT32_BYTE = 4;
// SortAndTopk的阈值，排序轴大于该阈值的场景，走sortAndTopK模板
const uint32_t SORT_AND_TOP_K_THRESHOLD = 10000000;
const uint32_t CONST_SIMT_SPACE = 32768; // 获取到的UB大小需要预留32KB给simt
const uint32_t SUPPORT_SORT_MAX_BYTE_SIZE = 8000;
const uint32_t SUPPORT_SORT_MAX_SIZE = 2000;
const uint32_t TOPK_MERGE_SORT_MORE_CORE_TILING_KEY_FLOAT = 23003;
const uint32_t TOPK_MERGE_SORT_INTRA_CORE_TILING_KEY_FLOAT = 33003;
const double FP32_K_LAST_AXIS_LOWER_RATIO = 0.25;
const double FP32_K_LAST_AXIS_UPPER_RATIO = 0.50;
const uint32_t SORT_STRUCT_SIZE_FP32 = 8;
const uint32_t FP32_MERGE_SORT_MAX_SIZE = 4096;
const int64_t NON_LAST_SMALL_AXIS_THRESHOLD = 2048;
const uint32_t MERGE_SORT_DISABLE_DOUBLE_BUFFER_SIZE = 2048;
const uint32_t MERGE_MORE_CORE_ONE_CORE_DATA_SIZE = 2048;
const uint32_t MERGE_MORE_CORE_LIST_MAX_NUM = 4;
const uint32_t MERGE_INTRA_CORE_SORT_ALIGN = 32;
const uint32_t MERGE_INTRA_CORE_MAX_BLOCKS = 256;
const float LAST_LOOP_CORE_UTILIZATION = 0.7;
const uint32_t SMALL_LOOP_UPPER_NUM = 4;
const uint32_t SMALL_LOOP_LOWER_NUM = 2;
const uint32_t SIMT_UB = 32768;       // SortAndTopK模板需要预留32k给simt使用
const uint32_t UB_CONST_INT32 = 4096; // 输出idx为int32时kernel侧需要的固定ub大小
const uint32_t UB_CONST_INT64 = 7168; // 输出idx为int64时kernel侧需要的固定ub大小
const uint32_t CONST_10 = 10;
const uint32_t CONST_14 = 14;
const uint32_t CONST_2 = 2;
const uint32_t SMALL_TILE_DATA_NUM = 1024; // 测试数据得出一次至少处理1024，sort性能比较好
// 排序轴在int32范围内的最大值, 超过这个值, cutsum，前缀和就要用int64数据范围表示
const uint32_t INT32_MAX_RANGE_VALUE_FOR_SORT = 1073741823;
const uint32_t SORT_STRUCT_BYTES = 8;

constexpr size_t SYS_WORK_SPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
struct TopkTileInfo {
    uint32_t coreNumNeed = 0;
    int64_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
    uint32_t ubRealLoadDataNum = 0;
    uint32_t oneCoreRowNum = 1;
    uint32_t batchNumInUb = 1;
    uint32_t tailLoopBatchNum = 0;
    uint32_t tailBatchNum = 0;
    uint32_t tailTileNum = 0;
    int64_t topKOutLastAxisNum = 0;
    bool multiCoreBigModel = false;
};
struct SortTileInfo {
    uint32_t coreNumNeed = 0;
    int64_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
    uint32_t oneCoreRowNum = 1;
    uint32_t ubSize = 0;
    uint32_t blockUbSize = 0;
    uint32_t dtypeSize = 0;
    uint32_t y2DtypeSize = 0;
    uint32_t maxCoreNum = 0;
    uint32_t numTileDataSize = 0;
    uint64_t sortLoopTimes = 0;
    uint32_t lastDimNeedCore = 0;
    uint32_t keyParams0 = 0;
    uint32_t keyParams1 = 0;
    uint32_t keyParams2 = 0;
    uint32_t keyParams3 = 0;
    uint32_t keyParams4 = 0;
    uint32_t keyParams5 = 0;
    uint32_t tmpUbSize = 0;
    bool isDescend = false;
    ge::DataType dataType = ge::DT_UINT8;
    uint32_t isInt32 = 0;
    int32_t xDimNum = 0;
    int64_t sortAxisNum = 1;
    int64_t unSortDimNum = 1;
    int64_t topKRealValue = 0;
    uint32_t tileDataSize = 0;
    uint32_t blockTileNum = 0;
    uint32_t tailTileNum = 0;
};
struct TopkComputeNowTileSizeInfo {
    ge::DataType dataType;
    ge::DataType indicesDType;
    bool isLargest = true;
    bool isSort = true;
    bool isInInt32Range = true;
    int64_t lastAxisNum = 0;
    int64_t kValue = 0;
    uint32_t maxCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t ubBlockAlignSize = 0; // ub的对齐数值，当前为32
    uint64_t unsortedDimNum = 0;
};
struct NonLastSmallAxisTopkLayout {
    uint32_t inputRowBytes = 0;
    uint32_t axisRowBytes = 0;
    uint32_t valueRowBytes = 0;
    uint32_t indexRowBytes = 0;
};
static const std::map<ge::DataType, uint32_t> tilingDataTypeKeyMap = {
    {ge::DT_INT64, 1004},  {ge::DT_INT32, 1003},   {ge::DT_INT16, 1002},  {ge::DT_INT8, 1001},
    {ge::DT_UINT64, 2004}, {ge::DT_UINT32, 2003},  {ge::DT_UINT16, 2002}, {ge::DT_UINT8, 2001},
    {ge::DT_FLOAT, 3003},  {ge::DT_FLOAT16, 3002}, {ge::DT_BF16, 4002}};
static const std::map<ge::DataType, uint32_t> tilingDataTypeBitMap = {
    {ge::DT_INT64, 8},  {ge::DT_INT32, 4},   {ge::DT_INT16, 2},  {ge::DT_INT8, 1},
    {ge::DT_UINT64, 8}, {ge::DT_UINT32, 4},  {ge::DT_UINT16, 2}, {ge::DT_UINT8, 1},
    {ge::DT_FLOAT, 4},  {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};
static const std::map<ge::DataType, uint32_t> optDataTypeBitMap = {
    {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};
static const std::map<ge::DataType, uint32_t> b64DataTypeBitMap = {{ge::DT_INT64, 8}, {ge::DT_UINT64, 8}};
} // namespace topkV2DataInfo

struct TopkNonLastSmallAxisTileInfo {
    int64_t rank = 0;
    int64_t sortAxis = 0;
    int64_t lastAxis = 1;
    int64_t outerSize = 1;
    int64_t innerSize = 1;
    int64_t unsortedDim = 1;
    uint32_t dtypeSize = 0;
    uint32_t y2DtypeSize = 0;
    uint32_t maxCoreNum = 0;
    uint32_t blockUbSize = 0;
    uint32_t tmpUbSize = 0;
    ge::DataType dataType = ge::DT_UINT8;
    uint32_t inputRowBytes = 0;
    uint32_t valueAxisBytes = 0;
    uint32_t indexAxisBytes = 0;
    uint32_t outputIndexRowBytes = 0;
};

struct TopkNonLastSmallAxisCandidate {
    uint32_t innerChunk = 0;
    uint32_t innerLoopNum = 0;
    uint32_t activeCore = 0;
    uint64_t tileCount = 0;
    uint64_t peakUb = 0;
    uint32_t inputRowBytes = 0;
    uint32_t valueAxisBytes = 0;
    uint32_t indexAxisBytes = 0;
    uint32_t outputIndexRowBytes = 0;
};

// ==================== Helper Functions ====================

inline uint32_t GetDataTypeSize(ge::DataType dataType)
{
    return topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second;
}

inline bool IsDataType64Bit(ge::DataType dataType) { return topkV2DataInfo::b64DataTypeBitMap.count(dataType) != 0; }

inline uint32_t GetDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::TMP_DATA_NUM_B64 : topkV2DataInfo::TMP_DATA_NUM;
}

inline uint32_t GetSingleBlockModelDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::SINGLE_BLOCK_DATA_NUM_B64 :
                                       topkV2DataInfo::SINGLE_BLOCK_DATA_NUM;
}

inline uint32_t GetSingleCoreModelDefaultTileDataSize(ge::DataType dataType)
{
    return IsDataType64Bit(dataType) ? topkV2DataInfo::SINGLE_CORE_DATA_NUM_B64 : topkV2DataInfo::SINGLE_CORE_DATA_NUM;
}

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

/**
 * topk自身的radix多核基数模板，前提条件: lastAxisNum 的值小于1000万
 */
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

uint32_t ComputeTopkRadixMoreCoreTileData(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                          topkV2DataInfo::TopkComputeNowTileSizeInfo& computeNowTileSizeInfo)
{
    uint32_t xDtypeSize = GetDataTypeSize(computeNowTileSizeInfo.dataType);
    uint32_t indexDtypeSize = GetDataTypeSize(computeNowTileSizeInfo.indicesDType);
    uint32_t tileData = GetDefaultTileDataSize(computeNowTileSizeInfo.dataType);

    uint64_t runTimeNeedSpace = GetTopkMultiCoreRunTimeNeedSpace(
        computeNowTileSizeInfo.lastAxisNum, tileData, computeNowTileSizeInfo.maxCoreNum, xDtypeSize, indexDtypeSize,
        indexDtypeSize, computeNowTileSizeInfo.kValue);

    int64_t lastDimTileNum = Ops::Base::CeilDiv(static_cast<uint64_t>(computeNowTileSizeInfo.lastAxisNum),
                                                static_cast<uint64_t>(tileData));
    int64_t maxInputK = std::max(lastDimTileNum, computeNowTileSizeInfo.kValue);

    GetTopkApiTmpBufferSize(context, topkTilingData, tileData, maxInputK, computeNowTileSizeInfo.isLargest,
                            computeNowTileSizeInfo.dataType, computeNowTileSizeInfo.isSort, tileData);
    uint32_t topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();

    while (topkAcApiNeedBuffer + runTimeNeedSpace > computeNowTileSizeInfo.ubSizePlatForm) {
        if (tileData < topkV2DataInfo::BIN_NUM) {
            OP_LOGD("TopKV2TilingForAscendC", "tileData is less than BIN_NUM, cannot decrease further.");
            return 0U;
        }
        tileData -= topkV2DataInfo::BIN_NUM;

        lastDimTileNum = Ops::Base::CeilDiv(static_cast<uint64_t>(computeNowTileSizeInfo.lastAxisNum),
                                            static_cast<uint64_t>(tileData));
        maxInputK = std::max(lastDimTileNum, computeNowTileSizeInfo.kValue);
        GetTopkApiTmpBufferSize(context, topkTilingData, tileData, maxInputK, computeNowTileSizeInfo.isLargest,
                                computeNowTileSizeInfo.dataType, computeNowTileSizeInfo.isSort, tileData);

        topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
        runTimeNeedSpace = GetTopkMultiCoreRunTimeNeedSpace(
            computeNowTileSizeInfo.lastAxisNum, tileData, computeNowTileSizeInfo.maxCoreNum, xDtypeSize, indexDtypeSize,
            indexDtypeSize, computeNowTileSizeInfo.kValue);
    }

    OP_LOGI("TopKV2TilingForAscendC", "Multi-core tile: data=%u, apiBuffer=%u", tileData, topkAcApiNeedBuffer);
    return tileData;
}

// 判断尾loop的核数利用率是否达标
bool IsLastLoopCoreUtilizationSuccess(uint64_t unsortedDimNum, uint32_t tmpOneCoreRowNum, uint32_t maxCoreNum)
{
    uint64_t virUnsortedDimNeedCoreNum = (unsortedDimNum + tmpOneCoreRowNum - 1) / tmpOneCoreRowNum;
    uint64_t sortLoopTimes = (virUnsortedDimNeedCoreNum + maxCoreNum - 1) / maxCoreNum;
    // 最后一个loop剩余的未处理dim数
    uint32_t lastLoopDimNum = static_cast<uint32_t>(unsortedDimNum %
                                                    (static_cast<uint64_t>(maxCoreNum) * tmpOneCoreRowNum));
    // 最后一个loop剩余的未处理dim数需要多少核数
    uint32_t lastLoopDimNeedCoreNum = lastLoopDimNum / tmpOneCoreRowNum;
    // 没有尾loop
    if (lastLoopDimNum == 0) {
        return true;
    }
    // 最后一次loop剩余待处理的轴数量/每个核处理的dim要大于0.7，确保最后一个loop有超过一半的核在处理，尽可能提高利用率
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
    // 按照每个核处理默认的 tileData 来计算一个核最多能处理多少行
    oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
    // virUnsortedDimNeedCoreNum: 默认最少需要多少核数
    uint64_t virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;

    // 默认最少需要的核数如果少于总核数, 说明如果按照之前的逻辑，会有核空闲，此时就应该将 unsortedDimNum
    // 均摊到所有的核上处理，然后返回tileData
    if (virUnsortedDimNeedCoreNum < maxCoreNum) {
        oneCoreRowNum = (unsortedDimNum + maxCoreNum - 1) / maxCoreNum;
        oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
        virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
        tileData = oneCoreRowNum * bufferNum * aglinNum;
        tileData = std::min(tileData, tileMaxData - topkV2DataInfo::BIN_NUM);
        return tileData;
    }

    // 按照的默认tileData来计算需要的虚拟核数比总核数还多，说明tileData切分较小，没有最大限度利用最大处理tileMaxData数据能力
    // 在不大于tileMaxData的条件下需要适当增加
    while (virUnsortedDimNeedCoreNum >= maxCoreNum && topkV2DataInfo::BIN_NUM + tileData < tileMaxData) {
        tileData += topkV2DataInfo::BIN_NUM;
        oneCoreRowNum = (tileData / bufferNum) / aglinNum;
        oneCoreRowNum = (oneCoreRowNum == 0) ? 1 : oneCoreRowNum;
        virUnsortedDimNeedCoreNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
    }

    uint32_t tmpTileData = tileData;
    // 在经过上述操作后的tileData, 可能导致在多loop中, 前几个loop核数处理数据比较理想，但是存在最后一个loop
    // 的极少数尾轴只有1个或者少数几个核在处理，导致最后一个loop将整个处理时间拉长,此时需要平衡最后一个loop,
    // 将tileData适当减少，使得处理尾loop的核数占总核数达到一定比例(0.7)，确保每个loop都是均匀处理数据
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

uint32_t ComputeMergeSortTileData(TopKV2TilingDataSimd& topkTilingData,
                                  topkV2DataInfo::TopkComputeNowTileSizeInfo& computeTileSizeInfo)
{
    ge::DataType dataType = computeTileSizeInfo.dataType;
    ge::DataType indicesDType = computeTileSizeInfo.indicesDType;
    int64_t lastAxisNum = computeTileSizeInfo.lastAxisNum;
    uint32_t maxCoreNum = computeTileSizeInfo.maxCoreNum;
    uint64_t unsortedDimNum = computeTileSizeInfo.unsortedDimNum;
    uint64_t ubSizePlatForm = computeTileSizeInfo.ubSizePlatForm;

    uint32_t xDtypeSize = GetDataTypeSize(dataType);
    uint32_t indexToDtypeSize = GetDataTypeSize(indicesDType);
    uint32_t convertTypeSize = (dataType == ge::DT_BF16) ? GetDataTypeSize(ge::DT_FLOAT) : GetDataTypeSize(dataType);

    uint32_t mergeSortAcApiNeedBuffer = topkTilingData.get_mergSortAcApiNeedBufferSize();
    uint32_t initUb = ubSizePlatForm - mergeSortAcApiNeedBuffer;
    OP_LOGD("TopKV2TilingForAscendC",
            "merge sort mergeSortAcApiNeedBuffer=%u, ubSizePlatForm=%u, "
            "convertTypeSize=%u",
            mergeSortAcApiNeedBuffer, ubSizePlatForm, convertTypeSize);

    uint32_t aglinNum = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(lastAxisNum), topkV2DataInfo::AGLIN_FACTOR));
    uint32_t bufferNum = lastAxisNum >= topkV2DataInfo::MERGE_SORT_DISABLE_DOUBLE_BUFFER_SIZE ?
                             1 :
                             topkV2DataInfo::CONST_TWO;
    uint32_t initSpace = aglinNum * sizeof(uint32_t) +
                         aglinNum * topkV2DataInfo::CONST_TWO * convertTypeSize * topkV2DataInfo::INT64_BYTE;

    if (initUb < initSpace) {
        OP_LOGD("TopKV2TilingForAscendC",
                "Not enough remaining space, initUb=%u, tensorSpace=%u,"
                " bufferNum=%u",
                initUb, initSpace, bufferNum);
        // 空间不足返回默认最小tileData
        return topkV2DataInfo::SMALL_TILE_DATA_NUM;
    }

    uint32_t oneCoreRowNumSize = initUb - initSpace;
    // 每个核最多可以处理多少行
    uint32_t factor = topkV2DataInfo::CONST_TWO * xDtypeSize + indexToDtypeSize + convertTypeSize;
    uint32_t oneCoreNeedSpace = aglinNum * bufferNum * factor;
    uint32_t oneCoreRowNumMax = oneCoreRowNumSize / oneCoreNeedSpace;

    uint32_t tileMaxData = oneCoreRowNumMax * aglinNum * bufferNum;
    OP_LOGD("TopKV2TilingForAscendC",
            "tileMaxData=%u, maxCoreNum=%u, oneCoreRowNumMax=%d, "
            "oneCoreRowNumSize=%u",
            tileMaxData, maxCoreNum, oneCoreRowNumMax, oneCoreRowNumSize);

    uint32_t tileData = GetTileDataForMergeSort(unsortedDimNum, maxCoreNum, tileMaxData, bufferNum, aglinNum);

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

void TileModeSmallSizeOptim(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                            topkV2DataInfo::TopkTileInfo& topkTileInfo,
                            topkV2DataInfo::TopkComputeNowTileSizeInfo& computeTileSizeInfo)
{
    uint64_t unsortedDimNum = computeTileSizeInfo.unsortedDimNum;
    uint32_t maxCoreNum = computeTileSizeInfo.maxCoreNum;
    int64_t lastAxisNum = computeTileSizeInfo.lastAxisNum;
    ge::DataType dataType = computeTileSizeInfo.dataType;

    SetMergeSortTmpSize(context, dataType, lastAxisNum, topkTilingData);
    uint32_t nowTileSize = ComputeMergeSortTileData(topkTilingData, computeTileSizeInfo);

    uint32_t aglinNum = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(lastAxisNum), topkV2DataInfo::AGLIN_FACTOR));
    uint32_t bufferNum = lastAxisNum >= topkV2DataInfo::MERGE_SORT_DISABLE_DOUBLE_BUFFER_SIZE ?
                             1 :
                             topkV2DataInfo::CONST_TWO;

    uint32_t oneCoreRowNum = std::max((nowTileSize / bufferNum) / aglinNum, 1U);
    uint64_t virUnsortedDimNum = (unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum;
    uint64_t sortLoopTimes = (virUnsortedDimNum + maxCoreNum - 1) / maxCoreNum;

    uint32_t realCoreNum = virUnsortedDimNum % maxCoreNum;
    uint32_t coreNumNeed = (sortLoopTimes == 1) ? (realCoreNum == 0 ? maxCoreNum : realCoreNum) : maxCoreNum;

    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_lastDimTileNum(1);
    topkTilingData.set_unsortedDimParallel(coreNumNeed);
    topkTilingData.set_lastDimNeedCore(1);
    topkTilingData.set_numTileDataSize(lastAxisNum);
    topkTilingData.set_keyParams4(bufferNum);

    topkTileInfo.ubRealLoadDataNum = lastAxisNum;
    topkTileInfo.coreNumNeed = coreNumNeed;
    topkTileInfo.lastDimTileNum = 1;
    topkTileInfo.unsortedDimParallel = coreNumNeed;
    topkTileInfo.oneCoreRowNum = oneCoreRowNum;

    OP_LOGI("TopKV2TilingForAscendC",
            "Small size opt mode coreNumNeed=%u sortLoopTimes=%lu lastAxisNum=%ld, "
            "oneCoreRowNum=%u, nowTileSize=%u.",
            coreNumNeed, sortLoopTimes, lastAxisNum, oneCoreRowNum, nowTileSize);
}

uint64_t GetSingleBlockTopkRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize,
                                            uint32_t indexToDtypeSize, int64_t kValue)
{
    OP_CHECK_IF(lastAxisNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "lastAxisNum", std::to_string(lastAxisNum).c_str(),
                                                      "The value of lastAxisNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    // 若lastAxisNum > tileData，initUb返回为0. 当前函数外部不会对tileData, 那么之后lastAxisNum > tileData
    // 不会进入SingleBlock模板
    uint32_t batchNumInUb = tileData / lastAxisNum;
    // kernel侧实际以lastAxisNum计算空间，避免(1,1,1), dataType=int32无法处理
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

uint32_t ComputeSingleBlockTileData(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                    ge::DataType dataType, ge::DataType indicesDType, bool isLargest, bool isSort,
                                    int64_t lastAxisNum, int64_t kValue, uint64_t ubSizePlatForm)
{
    uint32_t xDtypeSize = GetDataTypeSize(dataType);
    uint32_t indexToDtypeSize = GetDataTypeSize(indicesDType);
    uint32_t tileData = GetSingleBlockModelDefaultTileDataSize(dataType);

    GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
    uint32_t topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
    uint64_t needSpace = GetSingleBlockTopkRunTimeNeedSpace(lastAxisNum, tileData, xDtypeSize, indexToDtypeSize,
                                                            kValue);

    while (topkAcApiNeedBuffer + needSpace > ubSizePlatForm) {
        tileData -= topkV2DataInfo::BIN_NUM;
        if (tileData < lastAxisNum) {
            OP_LOGI("TopKV2TilingForAscendC", "tileData is less than lastAxisNum, tileData=%u", tileData);
            return 0U;
        }
        GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
        topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
        needSpace = GetSingleBlockTopkRunTimeNeedSpace(lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue);
    }

    OP_LOGI("TopKV2TilingForAscendC", "single block model tileData=%u, TempBuffer=%lu, ApiTempBuffer=%u", tileData,
            needSpace, topkAcApiNeedBuffer);
    return tileData;
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

bool IsMultiCoreOptimMode(gert::TilingContext* context, uint32_t& inputNowTileSize,
                          TopKV2TilingDataSimd& topkTilingData,
                          topkV2DataInfo::TopkComputeNowTileSizeInfo& computeNowTileSizeInfo)
{
    uint32_t xDtypeSize = GetDataTypeSize(computeNowTileSizeInfo.dataType);
    uint32_t indexToDtypeSize = GetDataTypeSize(computeNowTileSizeInfo.indicesDType);
    uint32_t tileData = GetDefaultTileDataSize(computeNowTileSizeInfo.dataType);

    if (tileData < computeNowTileSizeInfo.kValue) {
        OP_LOGD("TopKV2TilingForAscendC", "K value exceeds initial tileData");
        return false;
    }
    GetTopkApiTmpBufferSize(context, topkTilingData, tileData, computeNowTileSizeInfo.kValue,
                            computeNowTileSizeInfo.isLargest, computeNowTileSizeInfo.dataType,
                            computeNowTileSizeInfo.isSort, tileData);
    uint32_t topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
    uint64_t needSpace = GetTopkMultiCoreOptimModeRunTimeNeedSpace(
        computeNowTileSizeInfo.lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, computeNowTileSizeInfo.kValue,
        computeNowTileSizeInfo.ubBlockAlignSize);
    OP_LOGD("TopKV2TilingForAscendC",
            "multi core optim model init tileData=%u, init tempBuffer=%lu, init apiTempBuffer=%u, xDtypeSize=%u, "
            "indexToDtypeSize=%u",
            tileData, needSpace, topkAcApiNeedBuffer, xDtypeSize, indexToDtypeSize);
    while (topkAcApiNeedBuffer + needSpace > computeNowTileSizeInfo.ubSizePlatForm) {
        tileData -= topkV2DataInfo::TILE_SIZE_DECREASING_FACTOR;
        if (tileData < computeNowTileSizeInfo.kValue) {
            OP_LOGD("TopKV2TilingForAscendC", "K value exceeds adjusted tileData");
            return false;
        }
        GetTopkApiTmpBufferSize(context, topkTilingData, tileData, computeNowTileSizeInfo.kValue,
                                computeNowTileSizeInfo.isLargest, computeNowTileSizeInfo.dataType,
                                computeNowTileSizeInfo.isSort, tileData);
        topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
        needSpace = GetTopkMultiCoreOptimModeRunTimeNeedSpace(computeNowTileSizeInfo.lastAxisNum, tileData, xDtypeSize,
                                                              indexToDtypeSize, computeNowTileSizeInfo.kValue,
                                                              computeNowTileSizeInfo.ubBlockAlignSize);
        OP_LOGD("TopKV2TilingForAscendC",
                "multi core optim model now tileData=%u, now tempBuffer=%lu, now apiTempBuffer=%u.", tileData,
                needSpace, topkAcApiNeedBuffer);
    }

    // 在确定正确的tileData之后，必须确保尾轴是多核模式，否则会出现多核的tiling模式，走的是singleBlock的模板
    if (tileData >= computeNowTileSizeInfo.lastAxisNum) {
        OP_LOGD("TopKV2TilingForAscendC", "tileData >= lastAxisNum, not suitable for multi-core");
        return false;
    }

    // Verify K * tileNum fits within tileData
    OP_CHECK_IF(
        tileData == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tileData", std::to_string(tileData).c_str(),
                                              "The value of tileData must be greater than 0."),
        return false);

    uint32_t lastDimTileNum = Ops::Base::CeilDiv(static_cast<uint64_t>(computeNowTileSizeInfo.lastAxisNum),
                                                 static_cast<uint64_t>(tileData));
    uint32_t inputTopkSize = computeNowTileSizeInfo.kValue * lastDimTileNum;

    if (inputTopkSize <= static_cast<uint32_t>(tileData)) {
        inputNowTileSize = tileData;
        OP_LOGI("TopKV2TilingForAscendC", "Multi-core optim valid: tileData=%u, topkSize=%u", inputNowTileSize,
                inputTopkSize);
        return true;
    }

    return false;
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

uint32_t ComputeSingleCoreTileData(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                   ge::DataType dataType, ge::DataType indicesDType, bool isLargest, bool isSort,
                                   int64_t lastAxisNum, int64_t kValue, uint64_t ubSizePlatForm)
{
    uint32_t xDtypeSize = GetDataTypeSize(dataType);
    uint32_t indexToDtypeSize = GetDataTypeSize(indicesDType);
    uint32_t tileData = GetSingleCoreModelDefaultTileDataSize(dataType);

    GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
    uint32_t topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
    uint64_t needSpace = GetSingleCoreTopkRunTimeNeedSpace(lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue,
                                                           isSort);

    while (topkAcApiNeedBuffer + needSpace > ubSizePlatForm) {
        tileData -= topkV2DataInfo::BIN_NUM;
        OP_CHECK_IF(
            tileData == 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tileData", std::to_string(tileData).c_str(),
                                                  "The value of tileData must be greater than 0."),
            return 0);
        GetTopkApiTmpBufferSize(context, topkTilingData, tileData, kValue, isLargest, dataType, isSort, tileData);
        topkAcApiNeedBuffer = topkTilingData.get_topkAcApiTmpBufferSize();
        needSpace = GetSingleCoreTopkRunTimeNeedSpace(lastAxisNum, tileData, xDtypeSize, indexToDtypeSize, kValue,
                                                      isSort);
        OP_LOGI("TopKV2TilingForAscendC", "single core model now tileData=%u, now TempBuffer=%lu, now ApiTempBuffer=%u",
                tileData, needSpace, topkAcApiNeedBuffer);
    }
    OP_LOGD("TopKV2TilingForAscendC", "single core model tileData=%u, TempBuffer=%lu, ApiTempBuffer=%u", tileData,
            needSpace, topkAcApiNeedBuffer);
    return tileData;
}

void TileModeSmallSize(uint64_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum,
                       TopKV2TilingDataSimd& topkTilingData, topkV2DataInfo::TopkTileInfo& topkTileInfo,
                       uint32_t nowTileSize)
{
    OP_CHECK_IF(lastAxisNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "lastAxisNum", std::to_string(lastAxisNum).c_str(),
                                                      "The value of lastAxisNum must be greater than 0."),
                return);
    // 能进入SingleBlock模板说明：tileData >= lastAxisNum;
    uint32_t batchNumInUb = nowTileSize / static_cast<uint32_t>(lastAxisNum);
    uint32_t batchNumSingleLoop = maxCoreNum * batchNumInUb;
    uint64_t sortLoopTimes = unsortedDimNum / batchNumSingleLoop;

    uint32_t tailBatchNumTotal = unsortedDimNum % batchNumSingleLoop;
    uint32_t tailBatchNumSingleCore = 0;
    uint32_t tailBatchNum = 0;
    uint32_t coreNumNeed = maxCoreNum;
    if (tailBatchNumTotal != 0) {
        tailBatchNumSingleCore = tailBatchNumTotal / maxCoreNum;
        tailBatchNum = tailBatchNumTotal % maxCoreNum;
        sortLoopTimes += 1;
    }
    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_unsortedDimParallel(coreNumNeed);
    topkTilingData.set_numTileDataSize(lastAxisNum);
    topkTilingData.set_lastDimTileNum(1);
    topkTilingData.set_lastDimNeedCore(1);
    topkTileInfo.batchNumInUb = batchNumInUb;
    topkTileInfo.tailLoopBatchNum = tailBatchNumSingleCore;
    topkTileInfo.tailBatchNum = tailBatchNum;

    topkTileInfo.ubRealLoadDataNum = lastAxisNum;
    topkTileInfo.coreNumNeed = coreNumNeed;
    topkTileInfo.lastDimTileNum = 1;
    topkTileInfo.unsortedDimParallel = coreNumNeed;
    OP_LOGI("TopKV2TilingForAscendC", "Small size mode coreNumNeed=%u sortLoopTimes=%lu lastAxisNum=%ld", coreNumNeed,
            sortLoopTimes, lastAxisNum);
}

void TileModeSingleCore(uint64_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum,
                        TopKV2TilingDataSimd& topkTilingData, topkV2DataInfo::TopkTileInfo& topkTileInfo,
                        uint32_t nowTileSize)
{
    OP_CHECK_IF(nowTileSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "nowTileSize", std::to_string(nowTileSize).c_str(),
                                                      "The value of nowTileSize must be greater than 0."),
                return);
    int64_t lastDimTileNum = (lastAxisNum + nowTileSize - 1) / nowTileSize;
    uint64_t sortLoopTimes = unsortedDimNum / maxCoreNum;
    uint32_t tailBatchNum = unsortedDimNum % maxCoreNum;
    uint32_t tileNum = lastAxisNum / lastDimTileNum;
    uint32_t tailTileNum = lastAxisNum % lastDimTileNum;

    topkTilingData.set_lastDimTileNum(lastDimTileNum);
    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_numTileDataSize(tileNum);
    topkTilingData.set_unsortedDimParallel(maxCoreNum);

    topkTilingData.set_modeType(topkV2DataInfo::SINGLE_CORE_MODE);

    topkTileInfo.tailBatchNum = tailBatchNum;
    topkTileInfo.tailTileNum = tailTileNum;

    topkTileInfo.ubRealLoadDataNum = tileNum + 1;
    topkTileInfo.coreNumNeed = maxCoreNum;
    topkTileInfo.lastDimTileNum = lastDimTileNum;
    topkTileInfo.unsortedDimParallel = maxCoreNum;

    OP_LOGI("TopKV2TilingForAscendC", "Single core mode coreNumNeed=%u sortLoopTimes=%lu lastAxisNum=%u",
            topkTileInfo.coreNumNeed, sortLoopTimes, lastAxisNum);
}

void TileModeMediumSize(uint64_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum,
                        TopKV2TilingDataSimd& topkTilingData, topkV2DataInfo::TopkTileInfo& topkTileInfo,
                        uint32_t nowTileSize)
{
    OP_CHECK_IF(nowTileSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "nowTileSize", std::to_string(nowTileSize).c_str(),
                                                      "The value of nowTileSize must be greater than 0."),
                return);
    // 能进入中规模radix基数排序，lastAxisNum范围在几十万到百万之间
    uint32_t lastDimTileNum = (static_cast<uint32_t>(lastAxisNum) + nowTileSize - 1) / nowTileSize;
    uint32_t unsortedDimParallel = maxCoreNum / lastDimTileNum;
    uint32_t coreNumNeed = lastDimTileNum * unsortedDimParallel;
    uint64_t sortLoopTimes = (unsortedDimNum + unsortedDimParallel - 1) / unsortedDimParallel;
    if (sortLoopTimes == 1) {
        coreNumNeed = unsortedDimNum * lastDimTileNum;
        unsortedDimParallel = unsortedDimNum;
    }
    topkTilingData.set_modeType(topkV2DataInfo::MULT_CORE_MODE);
    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_lastDimTileNum(lastDimTileNum);
    topkTilingData.set_unsortedDimParallel(unsortedDimParallel);
    topkTilingData.set_lastDimNeedCore(lastDimTileNum);
    topkTilingData.set_numTileDataSize(nowTileSize);
    topkTileInfo.ubRealLoadDataNum = nowTileSize;
    // 为了适配TopK拼接sortwithindex，TopK MediumSize Mode需要满核运行
    topkTileInfo.coreNumNeed = maxCoreNum;
    topkTileInfo.lastDimTileNum = lastDimTileNum;
    topkTileInfo.unsortedDimParallel = unsortedDimParallel;
    OP_LOGI("TopKV2TilingForAscendC", "Medium size mode coreNumNeed=%u sortLoopTimes=%lu lastAxisNum=%ld", coreNumNeed,
            sortLoopTimes, lastAxisNum);
    OP_LOGI("TopKV2TilingForAscendC", "Medium size mode lastDimTileNum=%u unsortedDimParallel=%u lastDimRealCore=%u",
            lastDimTileNum, unsortedDimParallel, lastDimTileNum);
}
void TileModeBigSize(uint64_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum,
                     TopKV2TilingDataSimd& topkTilingData, topkV2DataInfo::TopkTileInfo& topkTileInfo,
                     uint32_t nowTileSize)
{
    OP_CHECK_IF(nowTileSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopkV2", "nowTileSize", std::to_string(nowTileSize).c_str(),
                                                      "The value of nowTileSize must be greater than 0."),
                return);
    int64_t lastDimTileNum = (lastAxisNum + nowTileSize - 1) / nowTileSize;
    uint32_t coreNumNeed = static_cast<uint32_t>(std::min(static_cast<int64_t>(maxCoreNum), lastDimTileNum));
    topkTilingData.set_modeType(topkV2DataInfo::MULT_CORE_MODE);
    topkTilingData.set_sortLoopTimes(unsortedDimNum);
    topkTilingData.set_lastDimTileNum(lastDimTileNum);
    topkTilingData.set_unsortedDimParallel(1);
    topkTilingData.set_lastDimNeedCore(coreNumNeed);
    topkTilingData.set_numTileDataSize(nowTileSize);
    topkTileInfo.ubRealLoadDataNum = nowTileSize;
    // 为了适配TopK拼接sortwithindex，TopK  BigSize Mode需要满核运行
    topkTileInfo.coreNumNeed = maxCoreNum;
    topkTileInfo.lastDimTileNum = lastDimTileNum;
    topkTileInfo.unsortedDimParallel = 1;
    topkTileInfo.multiCoreBigModel = true; // topk radix大规模多核标志
    OP_LOGI("TopKV2TilingForAscendC", "Big size mode coreNumNeed=%u sortLoopTimes=%lu lastAxisNum=%ld", coreNumNeed,
            unsortedDimNum, lastAxisNum);
}

/**
 * Topk自身多核基数模板
 * 进入条件:
 * 1. 能计算出正确的tileSize；
 */
void TileTopkMoreCoreMode(uint64_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum,
                          TopKV2TilingDataSimd& topkTilingData, topkV2DataInfo::TopkTileInfo& topkTileInfo,
                          uint32_t nowTileSize)
{
    uint32_t sortedDimParallelData = (nowTileSize * maxCoreNum) / 2;
    if (lastAxisNum <= sortedDimParallelData) {
        TileModeMediumSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    } else {
        if (topkTileInfo.topKOutLastAxisNum > topkV2DataInfo::MAX_K_FOR_INT64) {
            nowTileSize /= topkV2DataInfo::CONST_TWO;
        }
        TileModeBigSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    }
}

/**
 * Topk自身多核基数排序优化模板
 * 进入条件:
 * 1. k * (lastAxisNum / nowTileSize) <= nowTileSize, 假设 lastAxisNum 一共需要N个核处理，每个核计算出Topk的值；
 *    然后将前面每个核的Topk的值集中到一个核（前提是这个核的UB能装的下），再进行一次Topk处理;
 */
void TileMultiCoreOptimSize(uint64_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum,
                            TopKV2TilingDataSimd& topkTilingData, topkV2DataInfo::TopkTileInfo& topkTileInfo,
                            uint32_t nowTileSize)
{
    uint32_t sortedDimParallelData = (nowTileSize * maxCoreNum) / 2;
    if (lastAxisNum <= sortedDimParallelData) {
        TileModeMediumSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    } else {
        TileModeBigSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    }
    // tiling和多核基数排序保持一致, modelType不同
    topkTilingData.set_modeType(topkV2DataInfo::MULT_CORE_OPTIM_MODE);
}

/**
 * 进入条件：
 * 1. 尾轴不超过1000万;
 * 2. topk自有的基于多核radix基数排序算能够计算出正确的UB tileSize;
 */
bool IsTopkRadixMoreCoreMode(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                             topkV2DataInfo::TopkComputeNowTileSizeInfo& computeNowTileSizeInfo, uint32_t& nowTileSize)
{
    if (computeNowTileSizeInfo.lastAxisNum >= topkV2DataInfo::SORT_AND_TOP_K_THRESHOLD) {
        OP_LOGD("[TopKV2Tiling] lastAxisNum exceeds 10 million.");
        return false;
    }
    uint32_t tmpTileSize = ComputeTopkRadixMoreCoreTileData(context, topkTilingData, computeNowTileSizeInfo);
    if (tmpTileSize > 0) {
        nowTileSize = tmpTileSize;
        OP_LOGD("[TopKV2Tiling]", "radix more core final tileSize=%u", nowTileSize);
        return true;
    }
    return false;
}

ge::graphStatus IsValidParam(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("TopKV2", "context"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDesc(0) == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDesc(1) == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "k"),
                return ge::GRAPH_FAILED);
    auto inputDataType = context->GetInputDesc(0)->GetDataType();
    auto kDataType = context->GetInputDesc(1)->GetDataType();
    OP_CHECK_IF(context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "outputShape"), return ge::GRAPH_FAILED);
    const gert::Shape outValueShape = context->GetOutputShape(0)->GetStorageShape();
    const gert::Shape outIndexShape = context->GetOutputShape(1)->GetStorageShape();
    OP_CHECK_IF(context->GetOutputDesc(0) == nullptr || context->GetOutputDesc(1) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "outputDesc"), return ge::GRAPH_FAILED);
    auto outputValueDataType = context->GetOutputDesc(0)->GetDataType();
    auto outputIndexDataType = context->GetOutputDesc(1)->GetDataType();
    OP_CHECK_IF(
        topkV2DataInfo::tilingDataTypeKeyMap.count(inputDataType) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(inputDataType).c_str(),
                                  "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16, BF16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputDataType != outputValueDataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x, values",
                    (Ops::Base::ToString(inputDataType) + ", " + Ops::Base::ToString(outputValueDataType)).c_str(),
                    "The dtypes of x and values must be the same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDataType != ge::DT_INT32 && kDataType != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "k", Ops::Base::ToString(kDataType).c_str(),
                                          "INT32 or INT64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputIndexDataType != ge::DT_INT32 && outputIndexDataType != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices",
                                          Ops::Base::ToString(outputIndexDataType).c_str(), "INT32 or INT64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outValueShape != outIndexShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "values, indices",
                    (Ops::Base::ToString(outValueShape) + ", " + Ops::Base::ToString(outIndexShape)).c_str(),
                    "The shapes of values and indices must be the same."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * 判断是否为 Single Block 模式
 *
 * 执行条件：
 * 1. lastAxisNum 能一次性装入 UB 中，即 tileSize >= lastAxisNum
 *    这样一个核可以处理多个 lastAxisNum（batch），充分利用 UB 空间
 * 2. 能计算出有效的 tileSize（大于 0）
 */
bool IsSingleBlockMode(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData, uint32_t& tileSize,
                       topkV2DataInfo::TopkComputeNowTileSizeInfo& computeTileSizeInfo)
{
    // 计算 Single Block 模式所需的 tileSize
    uint32_t computedTileSize = ComputeSingleBlockTileData(
        context, topkTilingData, computeTileSizeInfo.dataType, computeTileSizeInfo.indicesDType,
        computeTileSizeInfo.isLargest, computeTileSizeInfo.isSort, computeTileSizeInfo.lastAxisNum,
        computeTileSizeInfo.kValue, computeTileSizeInfo.ubSizePlatForm);

    // 判断是否为 Single Block 模式
    // 条件1：计算出的 tileSize 必须大于 0（有效）
    // 条件2：tileSize >= lastAxisNum（能一次性装入 UB）
    bool isSingleBlockMode = (computedTileSize > 0) && (computedTileSize >= computeTileSizeInfo.lastAxisNum);

    if (isSingleBlockMode) {
        tileSize = computedTileSize;
        OP_LOGI("[TopKV2Tiling]", "Single block mode enabled: tileSize=%u >= lastAxisNum=%ld", tileSize,
                computeTileSizeInfo.lastAxisNum);
    }
    return isSingleBlockMode;
}

/**
 * 执行条件：
 * 1. 数据类型bf16, float16, float32, 尾轴小于等于1024;
 */
bool IsSmallSizeMergeSortMode(ge::DataType dataType, int64_t lastAxisNum)
{
    bool isSmallMergeSort = lastAxisNum <= topkV2DataInfo::SMALL_MAX_DATA_SZIE &&
                            topkV2DataInfo::optDataTypeBitMap.count(dataType) != 0;
    return isSmallMergeSort;
}

/**
 * 执行条件：
 * 1. 非尾轴大于等于核数, B轴均匀分核确定是有性能收益, 均匀分核场景，需要考虑尾行处理的时间与核间同步时间的均衡；
 *    目前测试非均匀分核性能也有提升，故不区分是否均匀分核，直接返回true，后面如果有性能走这个模板有性能劣化可以考虑这一点;
 * 2. UB能找到合适的tileSize;
 */
bool IsSingleCoreMode(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData, uint32_t& initTileSize,
                      topkV2DataInfo::TopkComputeNowTileSizeInfo& computTileInfo)
{
    if (computTileInfo.unsortedDimNum < computTileInfo.maxCoreNum) {
        OP_LOGD("[TopKV2Tiling], single core model unsortedDimNum is less than maxCoreNum.");
        return false;
    }
    uint32_t nowTileSizeTmp = ComputeSingleCoreTileData(
        context, topkTilingData, computTileInfo.dataType, computTileInfo.indicesDType, computTileInfo.isLargest,
        computTileInfo.isSort, computTileInfo.lastAxisNum, computTileInfo.kValue, computTileInfo.ubSizePlatForm);
    if (nowTileSizeTmp > 0) {
        initTileSize = nowTileSizeTmp;
        OP_LOGD("[TopKV2Tiling]", "single core final tileSize=%u", initTileSize);
        return true;
    }
    return false;
}

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

void GetTopkMergeMoreCoreFp32(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData, uint32_t maxCoreNum,
                              uint64_t unsortedDimNum, int64_t lastAxisNum, int64_t outLastAxisNum,
                              uint32_t onceMaxElements, uint64_t ubSizePlatForm)
{
    uint32_t splitCoreNum = Ops::Base::CeilDiv(
        static_cast<uint64_t>(lastAxisNum), static_cast<uint64_t>(topkV2DataInfo::MERGE_MORE_CORE_ONE_CORE_DATA_SIZE));
    // 当前分支下lastAxisNum范围通常在10万以内
    uint32_t numTileDataSize = splitCoreNum == 0 ? 0 : static_cast<uint32_t>(lastAxisNum) / splitCoreNum;
    uint32_t coreNumNeed = unsortedDimNum * splitCoreNum;
    uint32_t onceMaxElementsAlign = (onceMaxElements / topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN) *
                                    topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN;
    int64_t lastDimTileNumTimes = Ops::Base::CeilDiv(static_cast<int64_t>(splitCoreNum),
                                                     static_cast<int64_t>(maxCoreNum));

    topkTilingData.set_modeType(topkV2DataInfo::FP32_MERGE_MORE_CORE_MODE);
    topkTilingData.set_sortLoopTimes(1);
    topkTilingData.set_unsortedDimParallel(unsortedDimNum);
    topkTilingData.set_lastDimNeedCore(splitCoreNum);
    topkTilingData.set_numTileDataSize(numTileDataSize);
    topkTilingData.set_lastDimTileNum(splitCoreNum);
    topkTilingData.set_oneCoreRowNum(1);
    topkTilingData.set_keyParams0(onceMaxElementsAlign);
    topkTilingData.set_lastAxisNum(lastAxisNum);
    topkTilingData.set_unsortedDimNum(unsortedDimNum);
    topkTilingData.set_topKRealValue(outLastAxisNum);
    topkTilingData.set_outputLastDimValue(outLastAxisNum);
    topkTilingData.set_lastDimTileNumTimes(lastDimTileNumTimes);

    uint32_t alignInput = AlignTopkMergeMoreCoreWorkspaceElems(lastAxisNum);
    size_t usrSize = static_cast<size_t>(unsortedDimNum) * alignInput * topkV2DataInfo::SORT_STRUCT_SIZE_FP32 *
                     topkV2DataInfo::CONST_TWO;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = usrSize + topkV2DataInfo::SYS_WORK_SPACE_SIZE;
    context->SetTilingKey(topkV2DataInfo::TOPK_MERGE_SORT_MORE_CORE_TILING_KEY_FLOAT);
    context->SetBlockDim(coreNumNeed);
    context->SetLocalMemorySize(ubSizePlatForm);
    context->SetScheduleMode(1);
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

ge::graphStatus TileModeFp32MoreCoreSort(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                         topkV2DataInfo::TopkComputeNowTileSizeInfo& computNowTileInfo)
{
    topkTilingData.set_isLargest(computNowTileInfo.isLargest);
    topkTilingData.set_isSort(computNowTileInfo.isSort);
    topkTilingData.set_isInInt32Range(computNowTileInfo.isInInt32Range);
    topkTilingData.set_platformCoreNum(computNowTileInfo.maxCoreNum);

    uint32_t mergeMoreCoreOnceMaxElements = ComputeTopkMergeMoreCoreOnceMaxElements(computNowTileInfo.ubSizePlatForm,
                                                                                    computNowTileInfo.indicesDType);

    GetTopkMergeMoreCoreFp32(context, topkTilingData, static_cast<uint32_t>(computNowTileInfo.maxCoreNum),
                             computNowTileInfo.unsortedDimNum, computNowTileInfo.lastAxisNum, computNowTileInfo.kValue,
                             mergeMoreCoreOnceMaxElements, computNowTileInfo.ubSizePlatForm);
    topkTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(topkTilingData.GetDataSize());
    OP_LOGI("TopKV2TilingForAscendC", "TopKV2 fp32 merge more-core tiling end");
    return ge::GRAPH_SUCCESS;
}

void GetTopkMergeIntraCoreFp32(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData, uint32_t maxCoreNum,
                               uint64_t unsortedDimNum, int64_t lastAxisNum, int64_t outLastAxisNum,
                               uint32_t blockSortSize, uint32_t extractChunkSize, uint64_t ubSizePlatForm)
{
    uint32_t blocksPerRow = Ops::Base::CeilDiv(static_cast<uint64_t>(lastAxisNum),
                                               static_cast<uint64_t>(blockSortSize));
    uint32_t alignNum = blocksPerRow * blockSortSize;
    uint32_t actualCoreNum = std::min(static_cast<uint64_t>(maxCoreNum), unsortedDimNum);
    int64_t batchPerCore = Ops::Base::CeilDiv(unsortedDimNum, static_cast<uint64_t>(actualCoreNum));
    uint32_t sortLoopTimes = 1;
    int64_t lastDimTileNumTimes = Ops::Base::CeilDiv(static_cast<int64_t>(blocksPerRow),
                                                     static_cast<int64_t>(maxCoreNum));

    topkTilingData.set_modeType(topkV2DataInfo::FP32_MERGE_INTRA_CORE_MODE);
    topkTilingData.set_sortLoopTimes(sortLoopTimes);
    topkTilingData.set_unsortedDimParallel(actualCoreNum);
    topkTilingData.set_lastDimNeedCore(actualCoreNum);
    topkTilingData.set_numTileDataSize(blockSortSize);
    topkTilingData.set_lastDimTileNum(blocksPerRow);
    topkTilingData.set_oneCoreRowNum(batchPerCore);
    topkTilingData.set_keyParams0(batchPerCore);
    topkTilingData.set_keyParams1(blockSortSize * topkV2DataInfo::CONST_TWO);
    topkTilingData.set_keyParams2(alignNum * topkV2DataInfo::CONST_TWO);
    topkTilingData.set_keyParams3(alignNum);
    topkTilingData.set_keyParams4(extractChunkSize);
    topkTilingData.set_keyParams5(
        blockSortSize == 0 ? 0 : static_cast<uint32_t>(std::numeric_limits<int32_t>::max() / blockSortSize));
    topkTilingData.set_lastAxisNum(lastAxisNum);
    topkTilingData.set_unsortedDimNum(unsortedDimNum);
    topkTilingData.set_topKRealValue(outLastAxisNum);
    topkTilingData.set_outputLastDimValue(outLastAxisNum);
    topkTilingData.set_lastDimTileNumTimes(lastDimTileNumTimes);

    size_t perCoreWorkspace = static_cast<size_t>(alignNum) * topkV2DataInfo::SORT_STRUCT_SIZE_FP32 *
                              topkV2DataInfo::CONST_TWO;
    size_t usrSize = perCoreWorkspace * actualCoreNum;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = usrSize + topkV2DataInfo::SYS_WORK_SPACE_SIZE;
    context->SetTilingKey(topkV2DataInfo::TOPK_MERGE_SORT_INTRA_CORE_TILING_KEY_FLOAT);
    context->SetBlockDim(actualCoreNum);
    context->SetLocalMemorySize(ubSizePlatForm);
    context->SetScheduleMode(1);
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

ge::graphStatus TileModeFp32IntraCoreSort(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                          topkV2DataInfo::TopkComputeNowTileSizeInfo& computNowTileInfo)
{
    topkTilingData.set_isLargest(computNowTileInfo.isLargest);
    topkTilingData.set_isSort(computNowTileInfo.isSort);
    topkTilingData.set_isInInt32Range(computNowTileInfo.isInInt32Range);
    topkTilingData.set_platformCoreNum(computNowTileInfo.maxCoreNum);

    uint32_t mergeIntraCoreBlockSortSize = ComputeTopkMergeIntraCoreBlockSortSize(computNowTileInfo.ubSizePlatForm);
    uint32_t mergeIntraCoreExtractChunkSize = ComputeTopkMergeIntraCoreExtractChunkSize(
        computNowTileInfo.ubSizePlatForm, computNowTileInfo.indicesDType);

    GetTopkMergeIntraCoreFp32(context, topkTilingData, static_cast<uint32_t>(computNowTileInfo.maxCoreNum),
                              computNowTileInfo.unsortedDimNum, computNowTileInfo.lastAxisNum, computNowTileInfo.kValue,
                              mergeIntraCoreBlockSortSize, mergeIntraCoreExtractChunkSize,
                              computNowTileInfo.ubSizePlatForm);
    topkTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(topkTilingData.GetDataSize());
    OP_LOGI("TopKV2TilingForAscendC", "TopKV2 fp32 merge intra-core tiling end");
    return ge::GRAPH_SUCCESS;
}

// sort核间模板tiling计算相关函数
uint32_t ComputeRemainUb(topkV2DataInfo::SortTileInfo& sortTileInfo, uint32_t tileData, uint32_t ubExtra,
                         uint32_t tileFactor)
{
    uint32_t tmpUb = sortTileInfo.ubSize - (ubExtra + tileFactor * tileData);
    return tmpUb;
}

void SetSortTmpSize(ge::DataType dataType, uint32_t tileData, bool isDescend,
                    topkV2DataInfo::SortTileInfo& sortTileInfo)
{
    int64_t realLen = std::min(sortTileInfo.sortAxisNum, static_cast<int64_t>(tileData));
    std::vector<int64_t> shapeVec = {realLen};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0, minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTiling", "api of sort shape is %ld, maxUb is %u", realLen, maxValue);
    sortTileInfo.tmpUbSize = maxValue;
    return;
}

void AdjTmpUb(topkV2DataInfo::SortTileInfo& sortTileInfo, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint32_t remainUbNew = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor) - sortTileInfo.tmpUbSize;
    remainUbNew = remainUbNew > sortTileInfo.blockUbSize ? (remainUbNew - sortTileInfo.blockUbSize) : uint32_t(0);
    uint32_t alignUbSize = (remainUbNew / sortTileInfo.blockUbSize) * sortTileInfo.blockUbSize;
    OP_LOGI("RadixSortTiling", "alignUbSize %u", alignUbSize);
    sortTileInfo.tmpUbSize = sortTileInfo.tmpUbSize + alignUbSize; // 剩余的ub都给tmpUbsize
}

void ComputeTileDataOne(topkV2DataInfo::SortTileInfo& sortTileInfo, int64_t lastDimTileNum, uint32_t ubExtra,
                        uint32_t& tileData, uint32_t tileFactor)
{
    uint32_t allCore = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(lastDimTileNum), static_cast<uint64_t>(sortTileInfo.maxCoreNum)));
    uint32_t newTileData = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(sortTileInfo.sortAxisNum), static_cast<uint64_t>(allCore)));
    tileData = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(newTileData), static_cast<uint64_t>(topkV2DataInfo::BIN_NUM)));
    tileData = std::max(tileData, topkV2DataInfo::SMALL_TILE_DATA_NUM);
    SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
    AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
    return;
}

bool NeedAdjTileData(topkV2DataInfo::SortTileInfo& sortTileInfo, uint32_t& tileData, int64_t lastDimTileNum,
                     uint32_t ubExtra, uint32_t tileFactor)
{
    if (sortTileInfo.unSortDimNum == int64_t(1) && lastDimTileNum == int64_t(1)) {
        OP_LOGI("RadixSortTiling", "unSortDimNum and lastDimTileNum is 1");
        uint32_t newTileData = static_cast<uint32_t>(Ops::Base::CeilDiv(
            static_cast<uint64_t>(sortTileInfo.sortAxisNum), static_cast<uint64_t>(sortTileInfo.maxCoreNum)));
        newTileData = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<uint64_t>(newTileData), static_cast<uint64_t>(topkV2DataInfo::BIN_NUM)));
        tileData = std::max(newTileData, topkV2DataInfo::SMALL_TILE_DATA_NUM);
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return true;
    }
    if (sortTileInfo.unSortDimNum == int64_t(1) || (lastDimTileNum >= sortTileInfo.maxCoreNum)) {
        // b为1时，尽量均匀分核，同时保证处理的最小的tile_data为1024
        OP_LOGI("RadixSortTiling", "unSortDimNum is 1 and lastDimTileNum greater than allCore");
        ComputeTileDataOne(sortTileInfo, lastDimTileNum, ubExtra, tileData, tileFactor);
        return true;
    }
    if (sortTileInfo.unSortDimNum > int64_t(1) && sortTileInfo.unSortDimNum < int64_t(sortTileInfo.maxCoreNum) &&
        lastDimTileNum == int64_t(1)) {
        OP_LOGI("RadixSortTiling", "unSortDimNum greater than 1,and unSortDimNum small and lastDimTileNum is one");
        uint32_t hCore = sortTileInfo.maxCoreNum / static_cast<uint32_t>(sortTileInfo.unSortDimNum);
        uint32_t hTileData = static_cast<uint32_t>(sortTileInfo.sortAxisNum) / hCore;
        tileData = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<uint64_t>(hTileData), static_cast<uint64_t>(topkV2DataInfo::BIN_NUM)));
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return tileData;
    }
    if (sortTileInfo.unSortDimNum > int64_t(1) && lastDimTileNum > int64_t(1)) {
        // b大于1且h轴循环次数小于总核数，也就是b轴核数大于1
        OP_LOGI("RadixSortTiling", "unSortDimNum is one, lastDimTileNum greater than one");
        int64_t newTileData = sortTileInfo.sortAxisNum / int64_t(lastDimTileNum);
        tileData = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<uint64_t>(newTileData), static_cast<uint64_t>(topkV2DataInfo::BIN_NUM)));
        lastDimTileNum = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(sortTileInfo.sortAxisNum), static_cast<uint64_t>(tileData)));
        uint32_t bCore = lastDimTileNum == 0 ? sortTileInfo.maxCoreNum : sortTileInfo.maxCoreNum / lastDimTileNum;
        if (lastDimTileNum < sortTileInfo.maxCoreNum && sortTileInfo.unSortDimNum < int64_t(sortTileInfo.maxCoreNum)) {
            if (sortTileInfo.unSortDimNum < int64_t(bCore)) {
                bCore = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
                uint32_t hCore = sortTileInfo.maxCoreNum / bCore;
                uint32_t tileDataNew = static_cast<uint32_t>(
                    Ops::Base::CeilDiv(static_cast<uint64_t>(sortTileInfo.sortAxisNum), static_cast<uint64_t>(hCore)));
                tileData = static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<uint64_t>(tileDataNew),
                                                                      static_cast<uint64_t>(topkV2DataInfo::BIN_NUM)));
            }
        }
        if (bCore == static_cast<uint32_t>(1) && lastDimTileNum < sortTileInfo.maxCoreNum) {
            ComputeTileDataOne(sortTileInfo, lastDimTileNum, ubExtra, tileData, tileFactor);
        }
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return true;
    }
    return false;
}

uint32_t ComputeTileData(topkV2DataInfo::SortTileInfo& sortTileInfo)
{
    uint32_t ubExtra;
    uint32_t tileFactor;
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        ubExtra = topkV2DataInfo::UB_CONST_INT64;
        tileFactor = topkV2DataInfo::CONST_14 + sortTileInfo.dtypeSize;
    } else {
        ubExtra = topkV2DataInfo::UB_CONST_INT32;
        tileFactor = topkV2DataInfo::CONST_10 + sortTileInfo.dtypeSize;
    }

    uint32_t tileData = (sortTileInfo.ubSize - ubExtra) / tileFactor;
    tileData = (tileData / topkV2DataInfo::BIN_NUM) * topkV2DataInfo::BIN_NUM;

    uint32_t remainUb = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor);
    SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);

    uint32_t tmpUbSize = sortTileInfo.tmpUbSize;
    while (tmpUbSize > remainUb) {
        tileData = tileData - topkV2DataInfo::BIN_NUM;
        remainUb = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor);
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        tmpUbSize = sortTileInfo.tmpUbSize;
    }
    int64_t lastDimTileNum = Ops::Base::CeilDiv(sortTileInfo.sortAxisNum, static_cast<int64_t>(tileData));
    OP_LOGI("RadixSortTiling", "tileData %u, lastDimTileNum %ld, tmpUbSize %u", tileData, lastDimTileNum, tmpUbSize);
    bool smallTile = (sortTileInfo.sortAxisNum <= static_cast<int64_t>(topkV2DataInfo::SMALL_TILE_DATA_NUM)) &&
                     lastDimTileNum == int64_t(1);
    if ((lastDimTileNum % sortTileInfo.maxCoreNum == static_cast<uint32_t>(0)) || smallTile) {
        OP_LOGI("RadixSortTiling", "lastDimTileNum align or smallTile");
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return tileData;
    }
    if (NeedAdjTileData(sortTileInfo, tileData, lastDimTileNum, ubExtra, tileFactor)) {
        return tileData;
    }
    AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
    return tileData;
}

void ComputeWorkSpace(gert::TilingContext* context, topkV2DataInfo::SortTileInfo& sortTileInfo)
{
    uint32_t dtypeSizeWk = static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        dtypeSizeWk = static_cast<uint32_t>(sizeof(int64_t));
    }
    size_t excusiveBinsGmWkSize = static_cast<size_t>(sortTileInfo.keyParams1) * sortTileInfo.keyParams4 * dtypeSizeWk;
    excusiveBinsGmWkSize = static_cast<size_t>(Ops::Base::CeilAlign(static_cast<uint64_t>(excusiveBinsGmWkSize),
                                                                    static_cast<uint64_t>(sortTileInfo.blockUbSize)));

    size_t globalHistGmWkSize = static_cast<size_t>(sortTileInfo.keyParams3) * sortTileInfo.keyParams2 *
                                sortTileInfo.keyParams0 * dtypeSizeWk;
    globalHistGmWkSize = static_cast<size_t>(Ops::Base::CeilAlign(static_cast<uint64_t>(globalHistGmWkSize),
                                                                  static_cast<uint64_t>(sortTileInfo.blockUbSize)));

    size_t outIdxDbWK = static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel * dtypeSizeWk;
    outIdxDbWK = static_cast<size_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(outIdxDbWK), static_cast<uint64_t>(sortTileInfo.blockUbSize)));

    size_t sortOutIdxGMWK = static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel *
                            sortTileInfo.y2DtypeSize;
    sortOutIdxGMWK = static_cast<size_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(sortOutIdxGMWK), static_cast<uint64_t>(sortTileInfo.blockUbSize)));

    size_t histTileGmWk = static_cast<size_t>(sortTileInfo.lastDimTileNum) * topkV2DataInfo::BIN_NUM *
                          sortTileInfo.unsortedDimParallel * sizeof(int16_t) * topkV2DataInfo::CONST_2;

    size_t xB8GmWkSize = static_cast<size_t>(sortTileInfo.lastDimTileNum) * sortTileInfo.numTileDataSize *
                         sortTileInfo.unsortedDimParallel;
    xB8GmWkSize = static_cast<size_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(xB8GmWkSize), static_cast<uint64_t>(sortTileInfo.blockUbSize)));

    size_t outValueDbWKSize = static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel *
                              sortTileInfo.dtypeSize;
    outValueDbWKSize = static_cast<size_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(outValueDbWKSize), static_cast<uint64_t>(sortTileInfo.blockUbSize)));
    outValueDbWKSize *= topkV2DataInfo::CONST_2;

    OP_LOGI("RadixSortTiling",
            "excusiveBinsGmWkSize %lu, globalHistGmWkSize %lu, outIdxDbWK %lu, sortOutIdxGMWK %lu, histTileGmWk %lu,"
            " xB8GmWkSize %lu, outValueDbWKSize %lu ",
            excusiveBinsGmWkSize, globalHistGmWkSize, outIdxDbWK, sortOutIdxGMWK, histTileGmWk, xB8GmWkSize,
            outValueDbWKSize);
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    size_t usrSize = excusiveBinsGmWkSize + globalHistGmWkSize + outIdxDbWK + sortOutIdxGMWK + histTileGmWk +
                     xB8GmWkSize + outValueDbWKSize;
    userWorkSpaceSize[0] = usrSize + topkV2DataInfo::SYS_WORK_SPACE_SIZE;
    return;
}

ge::graphStatus GetRadixSortMoreCore(gert::TilingContext* context, topkV2DataInfo::SortTileInfo& sortTileInfo)
{
    sortTileInfo.ubSize = sortTileInfo.ubSize - topkV2DataInfo::SIMT_UB;
    uint32_t tileData = ComputeTileData(sortTileInfo);
    uint32_t lastDimTileNum = Ops::Base::CeilDiv(static_cast<uint64_t>(sortTileInfo.sortAxisNum),
                                                 static_cast<uint64_t>(tileData));
    if (sortTileInfo.maxCoreNum <= lastDimTileNum) {
        sortTileInfo.unsortedDimParallel = static_cast<uint32_t>(1);
    } else {
        sortTileInfo.unsortedDimParallel = lastDimTileNum == 0 ? sortTileInfo.maxCoreNum :
                                                                 sortTileInfo.maxCoreNum / lastDimTileNum;
        if (sortTileInfo.unSortDimNum < static_cast<int64_t>(sortTileInfo.unsortedDimParallel)) {
            sortTileInfo.unsortedDimParallel = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
        }
    }
    sortTileInfo.numTileDataSize = tileData;
    sortTileInfo.sortLoopTimes = Ops::Base::CeilDiv(static_cast<uint64_t>(sortTileInfo.unSortDimNum),
                                                    static_cast<uint64_t>(sortTileInfo.unsortedDimParallel));
    sortTileInfo.lastDimNeedCore = std::min(sortTileInfo.maxCoreNum, lastDimTileNum);
    sortTileInfo.coreNumNeed = sortTileInfo.unsortedDimParallel * sortTileInfo.lastDimNeedCore;
    sortTileInfo.lastDimTileNum = lastDimTileNum;

    uint32_t ubSizeNum = sortTileInfo.tmpUbSize / static_cast<uint32_t>(sizeof(uint32_t));
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        ubSizeNum = sortTileInfo.tmpUbSize / static_cast<uint32_t>(sizeof(int64_t));
    }
    uint32_t allNumGloblHist = topkV2DataInfo::BIN_NUM * lastDimTileNum * sortTileInfo.dtypeSize *
                               sortTileInfo.unsortedDimParallel;
    uint32_t allNumExcusiveBin = topkV2DataInfo::BIN_NUM * sortTileInfo.dtypeSize * sortTileInfo.unsortedDimParallel;
    uint32_t oneCoreSize = Ops::Base::CeilDiv(static_cast<uint64_t>(allNumGloblHist),
                                              static_cast<uint64_t>(sortTileInfo.coreNumNeed));
    sortTileInfo.keyParams5 = std::max(static_cast<int64_t>(oneCoreSize),
                                       static_cast<int64_t>(sortTileInfo.blockUbSize));
    sortTileInfo.keyParams0 = Ops::Base::CeilDiv(static_cast<uint64_t>(allNumGloblHist),
                                                 static_cast<uint64_t>(sortTileInfo.keyParams5));
    sortTileInfo.keyParams3 = Ops::Base::CeilDiv(static_cast<uint64_t>(sortTileInfo.keyParams5),
                                                 static_cast<uint64_t>(ubSizeNum));
    sortTileInfo.keyParams2 = sortTileInfo.keyParams5 > ubSizeNum ? ubSizeNum : sortTileInfo.keyParams5;

    uint32_t oneCoreSize1 = Ops::Base::CeilDiv(static_cast<uint64_t>(allNumExcusiveBin),
                                               static_cast<uint64_t>(sortTileInfo.coreNumNeed));
    sortTileInfo.keyParams4 = std::max(static_cast<uint64_t>(oneCoreSize1),
                                       static_cast<uint64_t>(sortTileInfo.blockUbSize));

    sortTileInfo.keyParams1 = Ops::Base::CeilDiv(static_cast<uint64_t>(allNumExcusiveBin),
                                                 static_cast<uint64_t>(sortTileInfo.keyParams4));

    // 取前k个结果相关流程的tile计算
    uint32_t avilableUbSize = (sortTileInfo.ubSize - 1) / topkV2DataInfo::AGLIN_FACTOR * topkV2DataInfo::AGLIN_FACTOR;
    OP_CHECK_IF(avilableUbSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "avilableUbSize",
                                                      std::to_string(avilableUbSize).c_str(),
                                                      "The value of avilableUbSize must be greater than 0."),
                return ge::GRAPH_FAILED);
    auto dataType = context->GetInputDesc(0)->GetDataType();
    auto indicesDType = context->GetOutputDesc(1)->GetDataType();

    uint32_t xDtypeSize = GetDataTypeSize(dataType);
    uint32_t indexToDtypeSize = GetDataTypeSize(indicesDType);
    uint32_t kGetDtypeSize = std::max(xDtypeSize, indexToDtypeSize);
    OP_CHECK_IF(kGetDtypeSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "kGetDtypeSize",
                                                      std::to_string(kGetDtypeSize).c_str(),
                                                      "The value of kGetDtypeSize must be greater than 0."),
                return ge::GRAPH_FAILED);
    sortTileInfo.tileDataSize = avilableUbSize / kGetDtypeSize;
    uint32_t totalTileNum = (sortTileInfo.topKRealValue + sortTileInfo.tileDataSize - 1) / sortTileInfo.tileDataSize;
    sortTileInfo.blockTileNum = totalTileNum / sortTileInfo.maxCoreNum;
    sortTileInfo.tailTileNum = totalTileNum % sortTileInfo.maxCoreNum;
    OP_CHECK_IF(
        sortTileInfo.blockTileNum == 0 && sortTileInfo.tailTileNum == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "blockTileNum, tailTileNum",
            (std::to_string(sortTileInfo.blockTileNum) + ", " + std::to_string(sortTileInfo.tailTileNum)).c_str(),
            "The values of blockTileNum and tailTileNum cannot both be 0."),
        return ge::GRAPH_FAILED);

    ComputeWorkSpace(context, sortTileInfo);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputAndOutput(gert::TilingContext* context, topkV2DataInfo::SortTileInfo& sortTileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(topkV2DataInfo::SIMT_UB),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                                      "The value of ubSize must be greater than SIMT_UB."),
                return ge::GRAPH_FAILED);
    sortTileInfo.blockUbSize = Ops::Base::GetUbBlockSize(context);
    OP_LOGI(context->GetNodeName(), "ubSize is %ld, blockUbSize %u", ubSize, sortTileInfo.blockUbSize);
    sortTileInfo.ubSize = ubSize;
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(inputShapePtr->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    const gert::Shape& outShape = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());
    OP_CHECK_IF(
        inputShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "x, values",
            (std::to_string(inputShape.GetShapeSize()) + ", " + std::to_string(outShape.GetShapeSize())).c_str(),
            "The values of shape sizes of x and values must be positive."),
        return ge::GRAPH_FAILED);
    int32_t xDimNum = inputShape.GetDimNum();
    sortTileInfo.xDimNum = xDimNum;
    int64_t sortAxisNum = inputShape.GetDim(xDimNum - 1);
    sortTileInfo.sortAxisNum = sortAxisNum;
    int64_t unSortDimNum = static_cast<int64_t>(1);
    for (uint32_t i = 0; i < static_cast<uint32_t>((xDimNum - 1)); i++) {
        int64_t dimSize = static_cast<int64_t>(inputShape.GetDim(i));
        unSortDimNum *= dimSize;
    }
    sortTileInfo.unSortDimNum = unSortDimNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortCheckParams(gert::TilingContext* context, topkV2DataInfo::SortTileInfo& sortTileInfo)
{
    OP_CHECK_IF(CheckInputAndOutput(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CheckInputAndOutput", "GRAPH_FAILED",
                                                      "The value of CheckInputAndOutput must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    auto inputDescPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
    ge::DataType dataType = inputDescPtr->GetDataType();
    OP_CHECK_IF(
        topkV2DataInfo::tilingDataTypeBitMap.count(dataType) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                  "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16, BF16"),
        return ge::GRAPH_FAILED);
    sortTileInfo.dataType = dataType;
    sortTileInfo.dtypeSize = topkV2DataInfo::tilingDataTypeBitMap.find(dataType)->second;
    auto outDescPtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr);
    auto y2DType = outDescPtr->GetDataType();
    auto outDescPtr0 = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr0);
    auto y1DType = outDescPtr0->GetDataType();
    OP_CHECK_IF((y2DType != ge::DT_INT64) && (y2DType != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices", Ops::Base::ToString(y2DType).c_str(),
                                          "INT32 or INT64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(y1DType != dataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x, values",
                    (Ops::Base::ToString(dataType) + ", " + Ops::Base::ToString(y1DType)).c_str(),
                    "The dtypes of x and values must be the same."),
                return ge::GRAPH_FAILED);
    sortTileInfo.y2DtypeSize = topkV2DataInfo::tilingDataTypeBitMap.find(y2DType)->second;
    return ge::GRAPH_SUCCESS;
}

void FillTilingDataSort(gert::TilingContext* context, topkV2DataInfo::SortTileInfo& sortTileInfo,
                        TopKV2TilingDataSimd& topkTilingData)
{
    if (sortTileInfo.isDescend) {
        topkTilingData.set_isLargest(topkV2DataInfo::CONST_TWO);
    } else {
        topkTilingData.set_isLargest(topkV2DataInfo::CONST_ZERO);
    }
    topkTilingData.set_isInInt32Range(sortTileInfo.isInt32);
    topkTilingData.set_numTileDataSize(sortTileInfo.numTileDataSize);
    topkTilingData.set_unsortedDimParallel(sortTileInfo.unsortedDimParallel);
    topkTilingData.set_lastDimTileNum(sortTileInfo.lastDimTileNum);
    topkTilingData.set_sortLoopTimes(sortTileInfo.sortLoopTimes);
    topkTilingData.set_lastDimNeedCore(sortTileInfo.lastDimNeedCore);
    topkTilingData.set_keyParams0(sortTileInfo.keyParams0);
    topkTilingData.set_keyParams1(sortTileInfo.keyParams1);
    topkTilingData.set_keyParams2(sortTileInfo.keyParams2);
    topkTilingData.set_keyParams3(sortTileInfo.keyParams3);
    topkTilingData.set_keyParams4(sortTileInfo.keyParams4);
    topkTilingData.set_keyParams5(sortTileInfo.keyParams5);
    topkTilingData.set_tmpUbSize(sortTileInfo.tmpUbSize);
    topkTilingData.set_lastAxisNum(sortTileInfo.sortAxisNum);
    topkTilingData.set_unsortedDimNum(sortTileInfo.unSortDimNum);
    topkTilingData.set_topKRealValue(sortTileInfo.topKRealValue);
    topkTilingData.set_sortAndTopkTileDataSize(sortTileInfo.tileDataSize);
    topkTilingData.set_sortAndTopkBlockTileNum(sortTileInfo.blockTileNum);
    topkTilingData.set_sortAndTopkTailTileNum(sortTileInfo.tailTileNum);
    topkTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(topkTilingData.GetDataSize());
    return;
}

void PrintTilindDataSort(gert::TilingContext* context, topkV2DataInfo::SortTileInfo& sortTileInfo)
{
    OP_LOGI(context->GetNodeName(),
            "realCoreNum %u, numTileDataSize %u, unsortedDimParallel %u, "
            "lastDimTileNum %ld, sortLoopTimes %lu, lastDimNeedCore %u, keyParams0 %u, keyParams1 %u "
            "keyParams2 %u, keyParams3 %u, keyParams4 %u, keyParams5 %u, tmpUbSize %u, "
            "lastAxisNum %ld, unsortedDimNum %ld, topKRealValue %ld, tileDataSize %u, blockTileNum %u, tailTileNum %u",
            sortTileInfo.coreNumNeed, sortTileInfo.numTileDataSize, sortTileInfo.unsortedDimParallel,
            sortTileInfo.lastDimTileNum, sortTileInfo.sortLoopTimes, sortTileInfo.lastDimNeedCore,
            sortTileInfo.keyParams0, sortTileInfo.keyParams1, sortTileInfo.keyParams2, sortTileInfo.keyParams3,
            sortTileInfo.keyParams4, sortTileInfo.keyParams5, sortTileInfo.tmpUbSize, sortTileInfo.sortAxisNum,
            sortTileInfo.unSortDimNum, sortTileInfo.topKRealValue, sortTileInfo.tileDataSize, sortTileInfo.blockTileNum,
            sortTileInfo.tailTileNum);
    return;
}

ge::graphStatus TileModeSortAndTopK(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                    topkV2DataInfo::TopkComputeNowTileSizeInfo& computeNowTileSizeInfo)
{
    topkV2DataInfo::SortTileInfo sortTileInfo;

    OP_CHECK_IF(SortCheckParams(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SortCheckParams", "GRAPH_FAILED",
                                                      "The value of SortCheckParams must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    topkTilingData.set_modeType(topkV2DataInfo::SORT_AND_TOP_K_MODE);
    OP_LOGI("[TopKV2Tiling]", "topkTilingData.set_modeType is: %u, SORT_AND_TOP_K_MODE: %u",
            topkTilingData.get_modeType(), topkV2DataInfo::SORT_AND_TOP_K_MODE);
    sortTileInfo.maxCoreNum = static_cast<uint32_t>(computeNowTileSizeInfo.maxCoreNum);
    sortTileInfo.isDescend = static_cast<bool>(computeNowTileSizeInfo.isLargest);
    sortTileInfo.isInt32 = static_cast<uint32_t>(computeNowTileSizeInfo.lastAxisNum <=
                                                 topkV2DataInfo::INT32_MAX_RANGE_VALUE_FOR_SORT);
    sortTileInfo.topKRealValue = computeNowTileSizeInfo.kValue;
    OP_CHECK_IF(GetRadixSortMoreCore(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "GetRadixSortMoreCore", "GRAPH_FAILED",
                                                      "The value of GetRadixSortMoreCore must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    auto dataTypeKey = topkV2DataInfo::tilingDataTypeKeyMap.find(computeNowTileSizeInfo.dataType)->second;
    context->SetTilingKey(dataTypeKey);
    context->SetBlockDim(sortTileInfo.coreNumNeed);
    context->SetLocalMemorySize(sortTileInfo.ubSize);
    FillTilingDataSort(context, sortTileInfo, topkTilingData);
    PrintTilindDataSort(context, sortTileInfo);
    // sortAndTopK模板核心是Sort，不需要后续Topk相关的tiling计算过程
    OP_LOGI("TopKV2TilingForAscendC", "TopKV2Tiling end");
    return ge::GRAPH_SUCCESS;
}

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

/**
 * 执行条件：
 * 1. 仅处理float32数据类型, 4096 <= lastAxisNum <= 32768;
 * 2. splitCoreNum = ceil(lastAxisNum / 2048) > 1 且非尾轴数量 unsortedDimNum * splitCoreNum <= maxCoreNum 且 K > 0;
 */
bool IsTopkMergeSortMoreCoreFp32Mode(topkV2DataInfo::TopkComputeNowTileSizeInfo& computNowTileInfo)
{
    if (computNowTileInfo.dataType != ge::DT_FLOAT || computNowTileInfo.kValue <= 0) {
        return false;
    }
    double ratio = static_cast<double>(computNowTileInfo.kValue) / computNowTileInfo.lastAxisNum;
    if (ratio < topkV2DataInfo::FP32_K_LAST_AXIS_LOWER_RATIO || topkV2DataInfo::FP32_K_LAST_AXIS_UPPER_RATIO <= ratio) {
        return false;
    }

    uint32_t onceMaxElements = ComputeTopkMergeMoreCoreOnceMaxElements(computNowTileInfo.ubSizePlatForm,
                                                                       computNowTileInfo.indicesDType);
    if (computNowTileInfo.maxCoreNum == 0 || computNowTileInfo.unsortedDimNum == 0 ||
        onceMaxElements < topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN) {
        return false;
    }
    uint32_t splitCoreNum = Ops::Base::CeilDiv(
        static_cast<uint64_t>(computNowTileInfo.lastAxisNum),
        static_cast<uint64_t>(topkV2DataInfo::MERGE_MORE_CORE_ONE_CORE_DATA_SIZE));
    if (splitCoreNum <= 1 || splitCoreNum > computNowTileInfo.maxCoreNum) {
        return false;
    }
    if (static_cast<uint64_t>(computNowTileInfo.unsortedDimNum) * splitCoreNum <= computNowTileInfo.maxCoreNum) {
        OP_LOGD("TopKV2Tiling", "float32 more core model onceMaxElements = %d.", onceMaxElements);
        return true;
    }
    return false;
}

/**
 * 执行条件：
 * 1. MoreCore 路由未命中, 既 IsTopkMergeSortMoreCoreFp32Mode 条件不成立;
 * 2. dtype == fp32 且 尾轴 > 4096 且 blocksPerRow = ceil(N / blockSortSize) <= 256 且 K > 0
 *
 */
bool IsTopkMergeSortIntraCoreFp32Mode(topkV2DataInfo::TopkComputeNowTileSizeInfo& computTileInfo)
{
    if (computTileInfo.dataType != ge::DT_FLOAT || computTileInfo.unsortedDimNum < computTileInfo.maxCoreNum / 2) {
        return false;
    }
    if (computTileInfo.kValue <= 0) {
        return false;
    }
    double ratio = static_cast<double>(computTileInfo.kValue) / computTileInfo.lastAxisNum;
    if (ratio < topkV2DataInfo::FP32_K_LAST_AXIS_LOWER_RATIO || topkV2DataInfo::FP32_K_LAST_AXIS_UPPER_RATIO <= ratio) {
        return false;
    }
    if (!computTileInfo.isSort && computTileInfo.kValue <= topkV2DataInfo::SUPPORT_SORT_MAX_SIZE) {
        return false;
    }

    uint32_t blockSortSize = ComputeTopkMergeIntraCoreBlockSortSize(computTileInfo.ubSizePlatForm);
    uint32_t extractChunkSize = ComputeTopkMergeIntraCoreExtractChunkSize(computTileInfo.ubSizePlatForm,
                                                                          computTileInfo.indicesDType);
    if (blockSortSize == 0 || extractChunkSize == 0) {
        return false;
    }
    OP_LOGI("IsTopkMergeIntraCoreFp32", "blockSortSize = %d, extractChunkSize=%d, dataType=%d, kValue=%d.",
            blockSortSize, extractChunkSize, computTileInfo.kValue);
    uint64_t blocksPerRow = Ops::Base::CeilDiv(static_cast<uint64_t>(computTileInfo.lastAxisNum),
                                               static_cast<uint64_t>(blockSortSize));
    return blocksPerRow > 1 && blocksPerRow <= topkV2DataInfo::MERGE_INTRA_CORE_MAX_BLOCKS;
}

ge::graphStatus TileModeFp32MergeSort(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                      topkV2DataInfo::TopkComputeNowTileSizeInfo& computNowTileInfo)
{
    if (IsTopkMergeSortMoreCoreFp32Mode(computNowTileInfo)) {
        return TileModeFp32MoreCoreSort(context, topkTilingData, computNowTileInfo);
    } else {
        return TileModeFp32IntraCoreSort(context, topkTilingData, computNowTileInfo);
    }
}

bool IsFp32MergeSortMode(topkV2DataInfo::TopkComputeNowTileSizeInfo& computNowTileInfo)
{
    bool isMoreCoreModel = IsTopkMergeSortMoreCoreFp32Mode(computNowTileInfo);
    bool isIntraCoreModel = IsTopkMergeSortIntraCoreFp32Mode(computNowTileInfo);
    return isMoreCoreModel || isIntraCoreModel;
}

bool CeilAlignUint32(uint64_t rawSize, uint32_t alignSize, uint32_t& alignedSize)
{
    uint64_t result = Ops::Base::CeilAlign(rawSize, static_cast<uint64_t>(alignSize));
    if (result > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    alignedSize = static_cast<uint32_t>(result);
    return true;
}

uint32_t GetTopkPreferredInnerChunk(ge::DataType dataType, uint32_t index)
{
    static constexpr uint32_t CHUNK_CANDIDATES[][6] = {
        {4, 2, 1, 0, 0, 0},
        {8, 4, 2, 1, 0, 0},
        {16, 8, 4, 2, 1, 0},
        {32, 16, 8, 4, 2, 1},
    };
    static constexpr uint32_t CHUNK_VALID_COUNT[] = {3, 4, 5, 6};
    uint32_t group = 0;
    if (dataType == ge::DT_INT64 || dataType == ge::DT_UINT64) {
        group = 0;
    } else if (dataType == ge::DT_FLOAT || dataType == ge::DT_INT32 || dataType == ge::DT_UINT32) {
        group = 1;
    } else if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16 || dataType == ge::DT_INT16 ||
               dataType == ge::DT_UINT16) {
        group = 2;
    } else if (dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
        group = 3;
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

bool SearchTopkNonLastSmallAxisPlan(
    const TopkNonLastSmallAxisTileInfo& info, uint64_t usableUb,
    std::function<bool(TopkNonLastSmallAxisTileInfo&, uint32_t, uint64_t&, TopkNonLastSmallAxisCandidate&)> estimateUb,
    TopkNonLastSmallAxisCandidate& best, TopkNonLastSmallAxisTileInfo* selectedInfo = nullptr)
{
    for (uint32_t i = 0; i < 6; ++i) {
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

/**
 * @brief 初始化非尾轴小轴TopK的Tile信息并验证参数有效性
 * @param context Tiling上下文
 * @param inputShape 输入张量的形状
 * @param axis TopK操作的目标轴索引
 * @param computeInfo TopK计算信息
 * @param info 输出的Tile信息结构体
 * @return true 参数有效，false 参数无效
 */
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

/**
 * @brief 设置非尾轴小轴TopK的临时UB大小
 * @details 根据是否使用归并排序，查询并设置相应的临时UB大小
 * @param context Tiling上下文
 * @param topkTilingData TopK Tiling数据结构
 * @param computeInfo TopK计算信息
 * @param info Tile信息（会更新tmpUbSize字段）
 * @param useMergeSort 是否使用归并排序
 * @param sortCount 排序元素个数
 * @return ge::GRAPH_SUCCESS 成功设置，ge::GRAPH_FAILED 设置失败
 */
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

/**
 * @brief 搜索非尾轴小轴TopK的最优切分方案
 * @details 遍历不同的innerChunk大小，找到UB利用率最高的方案
 * @param context Tiling上下文
 * @param info Tile信息
 * @param computeInfo TopK计算信息
 * @param useMergeSort 是否使用归并排序
 * @param best 输出的最优候选方案
 * @return ge::GRAPH_SUCCESS 成功找到方案，ge::GRAPH_FAILED 未找到有效方案
 */
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

/**
 * @brief 填充非尾轴小轴TopK场景的TilingData并设置Context参数
 * @details 该函数将搜索得到的最优切分方案填充到topkTilingData中，并设置context的相关参数，
 *          包括TilingKey、BlockDim、LocalMemorySize等，最后保存TilingData并打印日志。
 */
ge::graphStatus FillTopkNonLastSmallAxisTilingData(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                                   const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                                   const TopkNonLastSmallAxisTileInfo& info,
                                                   const TopkNonLastSmallAxisCandidate& best, uint32_t dataTypeKey,
                                                   int32_t axis, bool useMergeSort, uint32_t sortCount,
                                                   uint64_t originUbSize)
{
    // 设置TopK的基本参数
    topkTilingData.set_isLargest(computeInfo.isLargest);
    topkTilingData.set_isSort(computeInfo.isSort);
    topkTilingData.set_isInInt32Range(computeInfo.isInInt32Range);
    topkTilingData.set_lastAxisNum(info.lastAxis);
    topkTilingData.set_unsortedDimNum(info.innerSize);
    topkTilingData.set_oneCoreRowNum(info.outerSize);
    topkTilingData.set_topKRealValue(computeInfo.kValue);
    topkTilingData.set_outputLastDimValue(computeInfo.kValue);
    topkTilingData.set_platformCoreNum(computeInfo.maxCoreNum);
    topkTilingData.set_sortLoopTimes(1);
    topkTilingData.set_unsortedDimParallel(best.activeCore);
    topkTilingData.set_lastDimTileNum(1);
    topkTilingData.set_lastDimNeedCore(1);
    topkTilingData.set_numTileDataSize(sortCount);
    topkTilingData.set_keyParams0(useMergeSort ? topkV2DataInfo::NON_LAST_SMALL_AXIS_MERGE_SORT :
                                                 topkV2DataInfo::NON_LAST_SMALL_AXIS_RADIX_SELECT);
    topkTilingData.set_keyParams1(best.innerChunk);
    topkTilingData.set_keyParams2(best.innerLoopNum);
    topkTilingData.set_keyParams3(best.inputRowBytes);
    topkTilingData.set_keyParams4(best.valueAxisBytes);
    topkTilingData.set_keyParams5(best.indexAxisBytes);

    context->SetTilingKey(dataTypeKey);
    context->SetBlockDim(best.activeCore);
    context->SetLocalMemorySize(computeInfo.ubSizePlatForm);
    context->SetScheduleMode(1);
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = topkV2DataInfo::SYS_WORK_SPACE_SIZE;
    topkTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(topkTilingData.GetDataSize());
    OP_LOGI(context->GetNodeName(),
            "TopKV2 non-last small-axis selected axis=%d axisLen=%ld outer=%ld inner=%ld k=%ld algo=%u "
            "innerChunk=%u tileCount=%lu activeCore=%u peakUb=%lu usableUb=%lu originUb=%lu",
            axis, info.lastAxis, info.outerSize, info.innerSize, computeInfo.kValue,
            useMergeSort ? topkV2DataInfo::NON_LAST_SMALL_AXIS_MERGE_SORT :
                           topkV2DataInfo::NON_LAST_SMALL_AXIS_RADIX_SELECT,
            best.innerChunk, best.tileCount, best.activeCore, best.peakUb, computeInfo.ubSizePlatForm, originUbSize);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 非尾轴小轴TopK模式的Tiling主函数（优化版本，不超过50行）
 * @details 该函数通过调用多个辅助函数，完成非尾轴小轴TopK场景的Tiling参数计算。
 *          处理流程：
 *          1. 初始化Tile信息并验证参数有效性
 *          2. 确定算法模式（归并排序或基数选择）
 *          3. 设置临时UB大小
 *          4. 搜索最优切分方案
 *          5. 填充TilingData并设置Context参数
 *
 * @param context Tiling上下文
 * @param topkTilingData TopK Tiling数据结构
 * @param computeInfo TopK计算信息
 * @param inputShape 输入张量的形状
 * @param axis TopK操作的目标轴索引
 * @param dataTypeKey 数据类型对应的Tiling Key
 * @param originUbSize 原始UB大小
 * @return ge::GRAPH_SUCCESS 成功完成Tiling，ge::GRAPH_FAILED 处理失败
 */
ge::graphStatus TileModeNonLastSmallAxisTopK(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                             topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                             const gert::Shape& inputShape, int32_t axis, uint32_t dataTypeKey,
                                             uint64_t originUbSize)
{
    // 步骤1：初始化Tile信息并验证参数有效性
    TopkNonLastSmallAxisTileInfo info;
    if (!InitTopkNonLastSmallAxisInfo(context, inputShape, axis, computeInfo, info)) {
        return ge::GRAPH_FAILED;
    }

    // 步骤2：确定算法模式和排序元素个数
    topkTilingData.set_modeType(topkV2DataInfo::NON_LAST_SMALL_AXIS_MODE);
    bool useMergeSort = UseTopkNonLastMergeSort(computeInfo.dataType, static_cast<uint32_t>(info.lastAxis));
    uint32_t sortCount = Ops::Base::CeilAlign(static_cast<uint32_t>(info.lastAxis),
                                              topkV2DataInfo::MERGE_INTRA_CORE_SORT_ALIGN);

    // 步骤3：设置临时UB大小
    if (SetupTopkNonLastSmallAxisTmpUb(context, topkTilingData, computeInfo, info, useMergeSort, sortCount) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 步骤4：搜索最优切分方案
    TopkNonLastSmallAxisCandidate best;
    if (SearchBestTopkNonLastSmallAxisPlan(context, info, computeInfo, useMergeSort, best) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 步骤5：填充TilingData并设置Context参数
    return FillTopkNonLastSmallAxisTilingData(context, topkTilingData, computeInfo, info, best, dataTypeKey, axis,
                                              useMergeSort, sortCount, originUbSize);
}

bool IsNonLastDimTopKMode(int32_t dimValue, size_t inputDimNum)
{
    return dimValue != static_cast<int32_t>(inputDimNum) - 1;
}

ge::graphStatus TopKV2Tiling(gert::TilingContext* context, int32_t maxCoreNum)
{
    OP_LOGI("TopKV2TilingForAscendC", "TopKV2Tiling start");
    TopKV2TilingDataSimd topkTilingData;
    OP_CHECK_IF(IsValidParam(context) == ge::GRAPH_FAILED,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopKV2", "IsValidParam", "GRAPH_FAILED",
                                                      "The value of IsValidParam must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxCoreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TopKV2", "maxCoreNum", std::to_string(maxCoreNum).c_str(),
                                                      "The value of maxCoreNum must be greater than 0."),
                return ge::GRAPH_FAILED);

    const gert::Shape inputShape = context->GetInputShape(0)->GetStorageShape();
    auto dataType = context->GetInputDesc(0)->GetDataType();
    const gert::Shape outShape = context->GetOutputShape(0)->GetStorageShape();
    auto dataTypeKey = topkV2DataInfo::tilingDataTypeKeyMap.find(dataType)->second;
    auto indicesDType = context->GetOutputDesc(1)->GetDataType();
    std::string opType(context->GetNodeType());

    // check property
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isSorted = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, isSorted);
    OP_LOGI(context->GetNodeName(), "isSorted=%u", *isSorted);
    const int* dimValuePtr = attrs->GetAttrPointer<int>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimValuePtr);
    OP_LOGI(context->GetNodeName(), "dimValuePtr=%d", *dimValuePtr);
    const bool* isLargest = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, isLargest);
    OP_LOGI(context->GetNodeName(), "isLargest=%d", *isLargest);

    // check the indices_dtype attr and actual value of indices output
    const int* indicesDTypeValuePtr = attrs->GetAttrPointer<int>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDTypeValuePtr);
    OP_LOGI(context->GetNodeName(), "indicesDTypeValuePtr=%d, outPutIndexType=%ld.", *indicesDTypeValuePtr,
            static_cast<int64_t>(indicesDType));

    // 获取输入张量的维度数量
    size_t inputDimNum = inputShape.GetDimNum();

    // 验证 dim 参数的有效性（必须在 [-inputDimNum, inputDimNum-1] 范围内）
    int32_t dimMin = -static_cast<int32_t>(inputDimNum);
    int32_t dimMax = static_cast<int32_t>(inputDimNum) - 1;
    int32_t dimValue = *dimValuePtr;

    OP_CHECK_IF(dimValue < dimMin || dimValue > dimMax,
                OP_LOGE_WITH_INVALID_ATTR(
                    context->GetNodeName(), "dim", std::to_string(dimValue).c_str(),
                    (std::string("range [") + std::to_string(dimMin) + ", " + std::to_string(dimMax) + "]").c_str()),
                return ge::GRAPH_FAILED);
    if (dimValue < 0) {
        dimValue += static_cast<int32_t>(inputDimNum);
    }
    int64_t lastAxisNum = inputShape.GetDim(inputDimNum - 1);
    OP_CHECK_IF(
        lastAxisNum <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(inputShape).c_str(),
                                              "The last axis of x must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t unsortedDimNum = 1;
    for (uint32_t i = 0; i < (inputDimNum - 1); i++) {
        unsortedDimNum *= inputShape.GetDim(i);
    }
    size_t outputDimNum = outShape.GetDimNum();
    int64_t outLastAxisNum = outShape.GetDim(outputDimNum - 1);
    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    uint32_t isInInt32Range = static_cast<uint32_t>(lastAxisNum <= int32Max);
    // Get Platform Info
    uint64_t ubSizePlatForm = 0;
    uint64_t originUbSizePlatForm = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context));
    originUbSizePlatForm = ubSizePlatForm;
    OP_LOGI("[TopKV2Tiling]", "ubBlock size = : %u, originUbSizePlatForm=%d", ubBlock, originUbSizePlatForm);

    // 预留空间给SIMT
    ubSizePlatForm -= topkV2DataInfo::CONST_SIMT_SPACE;
    // 用于模板tilingSize计算
    topkV2DataInfo::TopkComputeNowTileSizeInfo computeNowTileSizeInfo;
    computeNowTileSizeInfo.isLargest = *isLargest;
    computeNowTileSizeInfo.isSort = *isSorted;
    computeNowTileSizeInfo.isInInt32Range = lastAxisNum <= int32Max;
    computeNowTileSizeInfo.lastAxisNum = lastAxisNum;
    computeNowTileSizeInfo.kValue = outLastAxisNum;
    computeNowTileSizeInfo.maxCoreNum = maxCoreNum;
    computeNowTileSizeInfo.ubSizePlatForm = ubSizePlatForm;
    computeNowTileSizeInfo.dataType = dataType;
    computeNowTileSizeInfo.indicesDType = indicesDType;
    computeNowTileSizeInfo.ubBlockAlignSize = ubBlock;
    computeNowTileSizeInfo.unsortedDimNum = unsortedDimNum;
    OP_LOGI("[TopKV2Tiling]",
            "computeNowTileSizeInfo isLargest: %u, isSort: %u, isInInt32Range: %u, lastAxisNum: %u, kValue: %u, "
            "maxCoreNum: %u, ubSizePlatForm: %u",
            computeNowTileSizeInfo.isLargest, computeNowTileSizeInfo.isSort, computeNowTileSizeInfo.isInInt32Range,
            computeNowTileSizeInfo.lastAxisNum, computeNowTileSizeInfo.kValue, computeNowTileSizeInfo.maxCoreNum,
            computeNowTileSizeInfo.ubSizePlatForm);

    topkV2DataInfo::TopkTileInfo topkTileInfo;
    topkTileInfo.topKOutLastAxisNum = outLastAxisNum;
    uint32_t nowTileSize = topkV2DataInfo::TMP_DATA_NUM;

    // 处理非尾轴的topk场景
    if (IsNonLastDimTopKMode(dimValue, inputDimNum)) {
        computeNowTileSizeInfo.lastAxisNum = inputShape.GetDim(dimValue);
        computeNowTileSizeInfo.kValue = outShape.GetDim(dimValue);
        computeNowTileSizeInfo.isInInt32Range = computeNowTileSizeInfo.lastAxisNum <= int32Max;
        return TileModeNonLastSmallAxisTopK(context, topkTilingData, computeNowTileSizeInfo, inputShape, dimValue,
                                            dataTypeKey, originUbSizePlatForm);
    }

    // 处理尾轴的场景
    if (IsSmallSizeMergeSortMode(dataType, lastAxisNum)) {
        TileModeSmallSizeOptim(context, topkTilingData, topkTileInfo, computeNowTileSizeInfo);
        dataTypeKey += topkV2DataInfo::MERGE_SORT_TILING_OFFSET;
    } else if (IsSingleBlockMode(context, topkTilingData, nowTileSize, computeNowTileSizeInfo)) {
        TileModeSmallSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    } else if (IsFp32MergeSortMode(computeNowTileSizeInfo)) {
        return TileModeFp32MergeSort(context, topkTilingData, computeNowTileSizeInfo);
    } else if (IsSingleCoreMode(context, topkTilingData, nowTileSize, computeNowTileSizeInfo)) {
        TileModeSingleCore(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    } else if (IsMultiCoreOptimMode(context, nowTileSize, topkTilingData, computeNowTileSizeInfo)) {
        // topk核间模板优化，修改modeType, tilling不变
        TileMultiCoreOptimSize(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    } else if (IsTopkRadixMoreCoreMode(context, topkTilingData, computeNowTileSizeInfo, nowTileSize)) {
        TileTopkMoreCoreMode(unsortedDimNum, maxCoreNum, lastAxisNum, topkTilingData, topkTileInfo, nowTileSize);
    } else {
        // sortAndTopk模板处理超大尾轴
        return TileModeSortAndTopK(context, topkTilingData, computeNowTileSizeInfo);
    }

    if (topkTilingData.get_modeType() == topkV2DataInfo::MULT_CORE_MODE &&
        topkTileInfo.topKOutLastAxisNum > topkV2DataInfo::MAX_K_FOR_INT64 && topkTileInfo.multiCoreBigModel) {
        OP_LOGD("[TopKV2Tiling] radix more core tileSize value halved.");
        nowTileSize /= topkV2DataInfo::CONST_TWO;
    }

    int64_t maxInputK = std::max(outLastAxisNum, topkTileInfo.lastDimTileNum);

    GetTopkApiTmpBufferSize(context, topkTilingData, topkTileInfo.ubRealLoadDataNum, maxInputK, *isLargest, dataType,
                            *isSorted, nowTileSize);
    int64_t lastDimTileNumTimesValue = (topkTileInfo.lastDimTileNum + maxCoreNum - 1) / maxCoreNum;
    OP_LOGD("[TopKV2Tiling]", "lastAxisNum: %ld, int32Max: %ld, isInInt32Range: %u, nowTileSize: %u", lastAxisNum,
            int32Max, isInInt32Range, nowTileSize);

    // fill the topkTilingData
    context->SetTilingKey(dataTypeKey);
    context->SetBlockDim(topkTileInfo.coreNumNeed);
    context->SetScheduleMode(1);
    topkTilingData.set_isLargest(*isLargest);
    topkTilingData.set_isSort(*isSorted);
    topkTilingData.set_isInInt32Range(isInInt32Range);
    topkTilingData.set_lastAxisNum(lastAxisNum);
    topkTilingData.set_unsortedDimNum(unsortedDimNum);
    topkTilingData.set_lastDimTileNumTimes(lastDimTileNumTimesValue);
    topkTilingData.set_platformCoreNum(maxCoreNum);
    topkTilingData.set_topKRealValue(outLastAxisNum);
    topkTilingData.set_oneCoreRowNum(topkTileInfo.oneCoreRowNum);
    topkTilingData.set_outputLastDimValue(outLastAxisNum);
    topkTilingData.set_batchNumInUb(topkTileInfo.batchNumInUb);
    topkTilingData.set_tailLoopBatchNum(topkTileInfo.tailLoopBatchNum);
    topkTilingData.set_tailBatchNum(topkTileInfo.tailBatchNum);
    topkTilingData.set_tailTileNum(topkTileInfo.tailTileNum);
    OP_LOGI(
        context->GetNodeName(),
        "TopK V2 tilingData tilingKey is %u, isLargest is %u, modelType is %u,"
        "isSorted is %u, lastAxisNum is %ld, unsortedDimNum is %lu, lastDimTileNumTimesValue is %ld, outLastAxisNum is "
        "%ld,"
        "oneCoreRowNum is %u, sortLoopTimes is %lu, lastDimTileNum is %ld, unsortedDimParallel is %u, modeType is %u,"
        "lastDimNeedCore is %u, numTileDataSize is %u, batchNumInUb is %u, tailLoopBatchNum is %u, tailBatchNum is %u,"
        "tailTileNum is %u, coreNumNeed is %u",
        dataTypeKey, topkTilingData.get_isLargest(), topkTilingData.get_modeType(), topkTilingData.get_isSort(),
        topkTilingData.get_lastAxisNum(), topkTilingData.get_unsortedDimNum(), topkTilingData.get_lastDimTileNumTimes(),
        topkTilingData.get_topKRealValue(), topkTilingData.get_oneCoreRowNum(), topkTilingData.get_sortLoopTimes(),
        topkTilingData.get_lastDimTileNum(), topkTilingData.get_unsortedDimParallel(), topkTilingData.get_modeType(),
        topkTilingData.get_lastDimNeedCore(), topkTilingData.get_numTileDataSize(), topkTilingData.get_batchNumInUb(),
        topkTilingData.get_tailLoopBatchNum(), topkTilingData.get_tailBatchNum(), topkTilingData.get_tailTileNum(),
        topkTileInfo.coreNumNeed);

    // TopKV2 Workspace 计算流程
    size_t usrSize = 0;

    uint64_t alginFactor = topkV2DataInfo::AGLIN_FACTOR;
    uint32_t modeType = topkTilingData.get_modeType();
    int64_t lastDimTileNum = topkTileInfo.lastDimTileNum;
    uint32_t unsortedDimParallel = topkTileInfo.unsortedDimParallel;

    // 提取公共变量：索引类型大小（根据数据范围选择 int32 或 int64）
    size_t indexTypeSize = isInInt32Range ? sizeof(int32_t) : sizeof(int64_t);

    // 根据 modeType 计算 Workspace 大小
    if (modeType == topkV2DataInfo::SINGLE_CORE_MODE) {
        usrSize = lastDimTileNum * topkV2DataInfo::BIN_NUM * unsortedDimParallel * indexTypeSize;
    } else if (modeType == topkV2DataInfo::MULT_CORE_OPTIM_MODE && isInInt32Range) {
        uint32_t xDtypeSize = GetDataTypeSize(dataType);
        uint32_t xDtypeSizeFactor = lastDimTileNum * unsortedDimParallel;
        // 计算临时排序结果空间（对齐）
        size_t tempSortResultNeedSize = Ops::Base::CeilAlign(
            static_cast<uint64_t>(xDtypeSize * xDtypeSizeFactor * outLastAxisNum), alginFactor);
        // 计算临时排序索引空间（对齐）
        size_t tempSortIndexNeedSize = Ops::Base::CeilAlign(
            static_cast<uint64_t>(sizeof(int32_t) * xDtypeSizeFactor * outLastAxisNum), alginFactor);
        usrSize = tempSortResultNeedSize + tempSortIndexNeedSize;
    } else {
        size_t dataSetCumSumHistNeedSize = topkV2DataInfo::BIN_NUM * indexTypeSize * unsortedDimParallel;
        size_t dataSetTileTopkNeedSize = Ops::Base::CeilAlign(
            static_cast<uint64_t>(lastDimTileNum * indexTypeSize * unsortedDimParallel), alginFactor);
        usrSize = dataSetCumSumHistNeedSize + dataSetTileTopkNeedSize * topkV2DataInfo::CONST_TWO;
    }

    // sortWithIndex tiling&workspace计算流程，
    if (needSortWithIndex(topkTilingData, *isSorted, dataType)) {
        OP_CHECK_IF(
            sortWithIndex::RadixSortTilingOfIdx(context, topkTilingData, maxCoreNum, &usrSize) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "RadixSortTilingOfIdx", "GRAPH_FAILED",
                                                  "The value of RadixSortTilingOfIdx must be GRAPH_SUCCESS."),
            return ge::GRAPH_FAILED);
    }

    // save tilingdata
    topkTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(topkTilingData.GetDataSize());

    // set userWorkSpaceSize
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = usrSize + topkV2DataInfo::SYS_WORK_SPACE_SIZE;
    OP_LOGI("[TopKV2Tiling]", "user & system WorkSpace Size is : %lu", userWorkSpaceSize[0]);
    context->SetLocalMemorySize(ubSizePlatForm);
    context->SetScheduleMode(1);
    OP_LOGI("TopKV2TilingForAscendC", "TopKV2Tiling end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForTopKV2(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "AscendC Tiling starting GRAPH_SUCCESS");
    auto compileInfo = context->GetCompiledInfo<TopKV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum",
                                                      std::to_string(compileInfo->coreNum).c_str(),
                                                      "The value of coreNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4TopKV2(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const TopKV2CompileInfo*>(context->GetCompileInfo());
    OP_LOGI(context->GetNodeName(), "AscendC topk simd tiling");
    OP_CHECK_IF(TopKV2Tiling(context, compile_info->coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "TopKV2Tiling", "GRAPH_FAILED",
                                                      "The value of TopKV2Tiling must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TopKV2).Tiling(Tiling4TopKV2).TilingParse<TopKV2CompileInfo>(TilingPrepareForTopKV2);
} // namespace topkV2
} // namespace optiling
