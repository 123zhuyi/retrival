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
 * \file top_k_v2_tiling_base.h
 * \brief top_k_v2 common tiling data structures and helpers
 */
#ifndef TOP_K_V2_TILING_BASE_H
#define TOP_K_V2_TILING_BASE_H

#include <functional>
#include <limits>
#include <map>
#include <vector>
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "top_k_v2_tiling_arch35.h"

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
const uint32_t BITONIC_SORT_TILING_OFFSET = 50000;
const uint32_t BITONIC_SMALL_TOPK_MIN_K = 2;
const uint32_t BITONIC_SMALL_TOPK_MAX_K = 32;
const uint64_t BITONIC_SMALL_TOPK_POLICY = 1;
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
const uint32_t FP32_MERGE_INTRA_CORE_MIN_CORE_NUM_DIVISOR = 2; // IntraCore模式要求unsortedDimNum >= maxCoreNum / 该值
const uint32_t MAX_INNER_CHUNK_CANDIDATES = 6;
// 按数据类型字节大小分组的索引，用于GetTopkPreferredInnerChunk选择chunk候选集
const uint32_t INNER_CHUNK_GROUP_8BYTE = 0; // INT64/UINT64
const uint32_t INNER_CHUNK_GROUP_4BYTE = 1; // FLOAT/INT32/UINT32
const uint32_t INNER_CHUNK_GROUP_2BYTE = 2; // FLOAT16/BF16/INT16/UINT16
const uint32_t INNER_CHUNK_GROUP_1BYTE = 3; // INT8/UINT8
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
inline const std::map<ge::DataType, uint32_t> tilingDataTypeKeyMap = {
    {ge::DT_INT64, 1004},  {ge::DT_INT32, 1003},   {ge::DT_INT16, 1002},  {ge::DT_INT8, 1001},
    {ge::DT_UINT64, 2004}, {ge::DT_UINT32, 2003},  {ge::DT_UINT16, 2002}, {ge::DT_UINT8, 2001},
    {ge::DT_FLOAT, 3003},  {ge::DT_FLOAT16, 3002}, {ge::DT_BF16, 4002}};
inline const std::map<ge::DataType, uint32_t> tilingDataTypeBitMap = {
    {ge::DT_INT64, 8},  {ge::DT_INT32, 4},   {ge::DT_INT16, 2},  {ge::DT_INT8, 1},
    {ge::DT_UINT64, 8}, {ge::DT_UINT32, 4},  {ge::DT_UINT16, 2}, {ge::DT_UINT8, 1},
    {ge::DT_FLOAT, 4},  {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};
inline const std::map<ge::DataType, uint32_t> optDataTypeBitMap = {
    {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};
inline const std::map<ge::DataType, uint32_t> b64DataTypeBitMap = {{ge::DT_INT64, 8}, {ge::DT_UINT64, 8}};
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

uint32_t GetDataTypeSize(ge::DataType dataType);
bool IsDataType64Bit(ge::DataType dataType);
uint32_t GetDefaultTileDataSize(ge::DataType dataType);
uint32_t GetSingleBlockModelDefaultTileDataSize(ge::DataType dataType);
uint32_t GetSingleCoreModelDefaultTileDataSize(ge::DataType dataType);

// ==================== Common Align Helpers ====================

bool CeilAlignUint32(uint64_t rawSize, uint32_t alignSize, uint32_t& alignedSize);

// ==================== FP32 MergeSort Helpers ====================

uint32_t AlignTopkMergeMoreCoreWorkspaceElems(int64_t elementNum);
uint32_t ComputeTopkMergeMoreCoreOnceMaxElements(uint64_t ubSizePlatForm, ge::DataType indicesDType);
uint32_t ComputeTopkMergeIntraCoreBlockSortSize(uint64_t ubSizePlatForm);
uint32_t ComputeTopkMergeIntraCoreExtractChunkSize(uint64_t ubSizePlatForm, ge::DataType indicesDType);

// ==================== NonLastSmallAxis Helpers ====================

uint32_t GetTopkPreferredInnerChunk(ge::DataType dataType, uint32_t index);
bool UseTopkNonLastMergeSort(ge::DataType dataType, uint32_t axisLen);
ge::DataType GetTopkNonLastSortDtype(ge::DataType dataType, bool useMergeSort);
uint32_t GetTopkNonLastSortDtypeSize(uint32_t dtypeSize, bool useMergeSort, ge::DataType dataType);
bool GetTopkNonLastSortTmpSize(ge::DataType dataType, uint32_t sortCount, bool useMergeSort, bool isDescend,
                               uint32_t& tmpUbSize);
void ComputeTopkAxisDimProducts(const gert::Shape& shape, int64_t axis, TopkNonLastSmallAxisTileInfo& info);

// ==================== TopK API Buffer Calculation ====================

ge::graphStatus GetTopkApiTmpBufferSize(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                        uint32_t needDataNum, int64_t kValue, bool isLargest, ge::DataType dtype,
                                        bool isSort, uint32_t nowTileSize);

// ==================== Runtime Space Calculation ====================

uint64_t GetTopkMultiCoreRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t maxCoreNum,
                                          uint32_t xDtypeSize, uint32_t indexToDtypeSize, uint32_t indexDtypeSize,
                                          int64_t kValue);
uint64_t GetSingleBlockTopkRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize,
                                            uint32_t indexToDtypeSize, int64_t kValue);
uint64_t GetTopkMultiCoreOptimModeRunTimeNeedSpace(int64_t lastAxisNum, uint32_t tileData, uint32_t xDtypeSize,
                                                   uint32_t indexToDtypeSize, int64_t kValue,
                                                   uint64_t ubBlockAlignSize);
uint64_t GetSingleCoreTopkRunTimeNeedSpace(int64_t lastAxisNum, uint32_t nowTileSize, uint32_t xDtypeSize,
                                           uint32_t indexToDtypeSize, int64_t kValue, bool isSort);

// ==================== MergeSort Helpers ====================

bool IsLastLoopCoreUtilizationSuccess(uint64_t unsortedDimNum, uint32_t tmpOneCoreRowNum, uint32_t maxCoreNum);
uint32_t GetTileDataForMergeSort(uint64_t unsortedDimNum, uint32_t maxCoreNum, uint32_t tileMaxData, uint32_t bufferNum,
                                 uint32_t aglinNum);
void SetMergeSortTmpSize(gert::TilingContext* context, ge::DataType dataType, int64_t lastAxisNum,
                         TopKV2TilingDataSimd& topkTilingData);

// ==================== Mode Judgment Functions ====================

bool needSortWithIndex(TopKV2TilingDataSimd& topkTilingData, bool isSorted, ge::DataType dataType);
bool IsBitonicSmallTopkMode(int64_t kValue, int64_t sortPolicy, bool isSort = true);

// ==================== NonLastSmallAxis Calculation Helpers ====================

bool SearchTopkNonLastSmallAxisPlan(
    const TopkNonLastSmallAxisTileInfo& info, uint64_t usableUb,
    std::function<bool(TopkNonLastSmallAxisTileInfo&, uint32_t, uint64_t&, TopkNonLastSmallAxisCandidate&)> estimateUb,
    TopkNonLastSmallAxisCandidate& best, TopkNonLastSmallAxisTileInfo* selectedInfo = nullptr);
bool ComputeTopkNonLastLayout(const TopkNonLastSmallAxisTileInfo& info, uint32_t kValue, uint32_t innerChunk,
                              bool useMergeSort, topkV2DataInfo::NonLastSmallAxisTopkLayout& layout);
bool EstimateTopkNonLastSmallAxisUb(TopkNonLastSmallAxisTileInfo& info, uint32_t kValue, uint32_t innerChunk,
                                    bool useMergeSort, uint64_t& peakUb, TopkNonLastSmallAxisCandidate& candidate);
bool InitTopkNonLastSmallAxisInfo(gert::TilingContext* context, const gert::Shape& inputShape, int32_t axis,
                                  const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                  TopkNonLastSmallAxisTileInfo& info);
ge::graphStatus SetupTopkNonLastSmallAxisTmpUb(gert::TilingContext* context, TopKV2TilingDataSimd& topkTilingData,
                                               const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                               TopkNonLastSmallAxisTileInfo& info, bool useMergeSort,
                                               uint32_t sortCount);
ge::graphStatus SearchBestTopkNonLastSmallAxisPlan(gert::TilingContext* context, TopkNonLastSmallAxisTileInfo& info,
                                                   const topkV2DataInfo::TopkComputeNowTileSizeInfo& computeInfo,
                                                   bool useMergeSort, TopkNonLastSmallAxisCandidate& best);

} // namespace topkV2
} // namespace optiling

#endif // TOP_K_V2_TILING_BASE_H
