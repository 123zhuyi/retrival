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
 * \file sort_with_index_tiling.cc
 * \brief sort_with_index ac tiling impl
 */
#include "sort_with_index_tiling.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_util.h"
#include "util/platform_util.h"

#include <string>

namespace optiling {
constexpr size_t WORK_SPACE_SIZE = 16777216; // 16 * 1024 * 1024;
const uint32_t BIN_NUM = 256;
const uint32_t TILE_DATA_NUM = 4096;
const uint32_t MEDIUM_TILE_DATA_NUM = 2048;
const uint32_t SMALL_TILE_DATA_NUM = 1024;
const uint32_t TILE_DATA_NUM_B64 = 2048;
const uint32_t TMP_UB = 1024; // 暂预留给sort高级api的大小
const uint32_t CONST_10 = 10; // int32索引时, 计算各种tensor的乘法因子
const uint32_t CONST_14 = 14; // int64索引时, 计算各种tensor的乘法因子
const uint32_t CONST_6 = 6;   // int64索引时, 计算各种tensor的乘法因子
const uint32_t CONST_1 = 1;
const uint32_t CONST_2 = 2;
const uint32_t INT64_BYTE = 8;
const uint32_t INT32_BYTE = 4;
const uint32_t SIMT_UB = 32768;            // 预留了32k给simt使用
const uint32_t NEED_UB_SIZE_BYTE = 221184; // 预留了32k给simt使用
const uint32_t SMALL_SORT_MAX_DATA_SIZE = 512;
const uint32_t AGLIN_VALUE = 32;
const uint32_t MERGE_SORT_TILING_OFFSET = 10000;
const uint32_t UB_CONST_INT32 = 4096; // 输出idx为int32时kernel侧需要的固定ub大小
const uint32_t UB_CONST_INT64 = 7168; // 输出idx为int64时kernel侧需要的固定ub大小
// 排序轴在int32范围内的最大值, 超过这个值, cutsum，前缀和就要用int64数据范围表示
const uint32_t INT32_MAX_RANGE_VALUE = 1073741823;
const uint32_t SMALL_SIZE_OPTIM_MODE = 0;
const uint32_t SMALL_SIZE_MODE = 1;
const uint32_t MULT_CORE_MODE = 2;
struct SortTileInfo {
    uint32_t coreNumNeed = 0;
    uint32_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
    uint32_t oneCoreRowNum = 1;
    uint32_t ubSize = 0;
    uint32_t blockUbSize = 0;
    uint32_t dtypeSize = 0;
    uint32_t y2DtypeSize = 0;
    uint32_t maxCoreNum = 0;
    uint32_t numTileDataSize = 0;
    uint32_t sortLoopTimes = 0;
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

uint32_t CeilDiv1(int64_t a, int64_t b)
{
    if (b == 0) {
        return static_cast<uint32_t>(a);
    }
    return static_cast<uint32_t>((a + b - 1) / b);
}

template <typename T>
auto CeilDivMul1(int64_t a, int64_t b) -> T const
{
    if (b == 0) {
        return static_cast<T>(a);
    }
    return static_cast<T>(((a + b - 1) / b) * b);
}

void SetSortTmpSizeOfIdx(ge::DataType dataType, int64_t lastAxisNum, uint32_t tileData, bool isDescend, bool hasIndex,
                         SortWithIndexTilingDataSimt& sortTilingData, SortTileInfo& sortTileInfo)
{
    int64_t reanLen = std::min(lastAxisNum, static_cast<int64_t>(tileData));
    std::vector<int64_t> shapeVec = {reanLen};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    // SortWithIndex and no need to cut axis
    config.hasSrcIndex = hasIndex && (reanLen == lastAxisNum);
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGW("[SortWithIndexTilingForAscendC]", "Allocal buffer element len = %lu ac sort api", reanLen);
    OP_LOGW("[SortWithIndexTilingForAscendC]", "Need tmp buffer %u byte for ac sort api", maxValue);
    sortTilingData.set_sortAcApiNeedBufferSize(maxValue);
    sortTileInfo.tmpUbSize = maxValue;
}

void SetMergeSortTmpSizeOfIdx(gert::TilingContext* context, ge::DataType dataType, int64_t lastAxisNum,
                              SortWithIndexTilingDataSimt& sortTilingData)
{
    uint32_t reanLen = 0;
    if ((lastAxisNum <= SMALL_SORT_MAX_DATA_SIZE) && (optDataTypeBitMap.count(dataType) != 0)) {
        reanLen = std::min(static_cast<uint32_t>(lastAxisNum), SMALL_SORT_MAX_DATA_SIZE);
    }
    uint32_t aglinDataSize = static_cast<uint32_t>((reanLen + AGLIN_VALUE - 1) / AGLIN_VALUE * AGLIN_VALUE);
    uint32_t dataTypeSize = (dataType == ge::DT_BF16) ? optDataTypeBitMap.find(ge::DT_FLOAT)->second :
                                                        optDataTypeBitMap.find(dataType)->second;
    auto platform_info = context->GetPlatformInfo();
    if (nullptr == platform_info) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platform_info");
    }
    auto plat = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t dataSizeNeed = AscendC::GetConcatTmpSize(plat, aglinDataSize, dataTypeSize);
    OP_LOGW("[SortWithIndexTilingForAscendC]", "Allocal buffer mergesort element len = %u ac sort api", reanLen);
    OP_LOGW("[SortWithIndexTilingForAscendC]", "Merge sort need tmp buffer %u byte for ac api", dataSizeNeed);
    sortTilingData.set_mergSortAcApiNeedBufferSize(dataSizeNeed);
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
}

void TileModeSmallSizeOptimOfIdx(uint64_t unsortedDimNum, uint32_t maxCoreNum, int64_t lastAxisNum, uint32_t tileData,
                                 SortTileInfo& sortTileInfo)
{
    uint32_t aglinNum = static_cast<uint32_t>((lastAxisNum + AGLIN_VALUE - 1) / AGLIN_VALUE * AGLIN_VALUE);
    uint32_t oneCoreRowNum = static_cast<uint32_t>((tileData / 2) / aglinNum);
    oneCoreRowNum = static_cast<uint32_t>(oneCoreRowNum == 0 ? 1 : oneCoreRowNum);
    uint32_t virUnsortedDimNum = static_cast<uint32_t>((unsortedDimNum + oneCoreRowNum - 1) / oneCoreRowNum);
    uint32_t coreNumNeed = 0;
    OP_CHECK_IF(maxCoreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("TileModeSmallSizeOptimOfIdx", "maxCoreNum",
                                                      std::to_string(maxCoreNum).c_str(),
                                                      "The value of maxCoreNum must be greater than 0."),
                return);
    uint32_t sortLoopTimes = static_cast<uint32_t>((virUnsortedDimNum + maxCoreNum - 1) / maxCoreNum);
    if (sortLoopTimes == 1u) {
        uint32_t realCoreNum = virUnsortedDimNum % maxCoreNum;
        if (realCoreNum == 0u) {
            realCoreNum = maxCoreNum;
        }
        coreNumNeed = realCoreNum;
    } else {
        coreNumNeed = maxCoreNum;
    }
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.lastDimTileNum = 1U;
    sortTileInfo.unsortedDimParallel = coreNumNeed;
    sortTileInfo.oneCoreRowNum = oneCoreRowNum;
    sortTileInfo.lastDimNeedCore = 1;
    sortTileInfo.sortLoopTimes = sortLoopTimes;
    sortTileInfo.numTileDataSize = lastAxisNum;
    OP_LOGI("[SortWithIndexTilingForAscendC]",
            "Small size opt mode coreNumNeed=%u, sortLoopTimes=%u, lastAxisNum=%ld, "
            "oneCoreRowNum=%ld, ubsize=%lu.",
            coreNumNeed, sortLoopTimes, lastAxisNum, oneCoreRowNum, sortTileInfo.ubSize);
}

void TileModeSmallSizeOfIdx(gert::TilingContext* context, uint64_t unsortedDimNum, uint32_t maxCoreNum,
                            int64_t lastAxisNum, SortTileInfo& sortTileInfo)
{
    uint32_t coreNumNeed = 0;
    OP_CHECK_IF(
        maxCoreNum == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "maxCoreNum", std::to_string(maxCoreNum).c_str(),
                                              "The value of maxCoreNum must be greater than 0."),
        return);
    uint32_t sortLoopTimes = static_cast<uint32_t>((unsortedDimNum + maxCoreNum - 1) / maxCoreNum);
    if (sortLoopTimes == 1u) {
        uint32_t realCoreNum = unsortedDimNum % maxCoreNum;
        if (realCoreNum == 0u) {
            realCoreNum = maxCoreNum;
        }
        coreNumNeed = realCoreNum;
    } else {
        coreNumNeed = maxCoreNum;
    }
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    sortTileInfo.unsortedDimParallel = coreNumNeed;
    sortTileInfo.lastDimNeedCore = 1;
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(lastAxisNum);
    sortTileInfo.sortLoopTimes = sortLoopTimes;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    OP_LOGI("[SortWithIndexTilingForAscendC]",
            "Small size mode coreNumNeed=%u sortLoopTimes=%u "
            "lastAxisNum=%ld",
            coreNumNeed, sortLoopTimes, lastAxisNum);
}

void PrintTilingDataOfIdx(SortTileInfo& sortTileInfo, SortWithIndexTilingDataSimt& sortTilingData)
{
    OP_LOGI("[Print SortWithIndexTilingForAscendC TilingData]",
            "coreNum is %u, lastAxisNum is %ld, isInInt32Range is %u, "
            "sortLoopTimes is %u, unsortedDimParallel is %u, unsortedDimNum is %u, "
            "lastDimTileNum is %u, lastDimNeedCore is %u, numTileDataSize is %u, "
            "sortAcApiNeedBufferSize is %u, mergSortAcApiNeedBufferSize is %u, "
            "oneCoreRowNum is %u, outputLastDimValue is %u, tmp ub size is %u, "
            "keyParams0 is %u, keyParams1 is %u, keyParams2 is %u, keyParams3 is %u, keyParams4 is %u, "
            "keyParams5 is %u, ub avalibal size=%lu, modeType=%u.",
            sortTileInfo.coreNumNeed, sortTilingData.get_lastAxisNum(), sortTilingData.get_isInInt32Range(),
            sortTilingData.get_sortLoopTimes(), sortTilingData.get_unsortedDimParallel(),
            sortTilingData.get_unsortedDimNum(), sortTilingData.get_lastDimTileNum(),
            sortTilingData.get_lastDimNeedCore(), sortTilingData.get_numTileDataSize(),
            sortTilingData.get_sortAcApiNeedBufferSize(), sortTilingData.get_mergSortAcApiNeedBufferSize(),
            sortTilingData.get_oneCoreRowNum(), sortTilingData.get_outputLastDimValue(), sortTilingData.get_tmpUbSize(),
            sortTilingData.get_keyParams0(), sortTilingData.get_keyParams1(), sortTilingData.get_keyParams2(),
            sortTilingData.get_keyParams3(), sortTilingData.get_keyParams4(), sortTilingData.get_keyParams5(),
            sortTileInfo.ubSize, sortTilingData.get_modeType());
    return;
}

void FillRadixSortTilingDataSort(SortTileInfo& sortTileInfo, SortWithIndexTilingDataSimt& sortTilingData)
{
    sortTilingData.set_numTileDataSize(sortTileInfo.numTileDataSize);
    sortTilingData.set_unsortedDimParallel(sortTileInfo.unsortedDimParallel);
    sortTilingData.set_lastDimTileNum(sortTileInfo.lastDimTileNum);
    sortTilingData.set_sortLoopTimes(sortTileInfo.sortLoopTimes);
    sortTilingData.set_lastDimNeedCore(sortTileInfo.lastDimNeedCore);
    sortTilingData.set_keyParams0(sortTileInfo.keyParams0);
    sortTilingData.set_keyParams1(sortTileInfo.keyParams1);
    sortTilingData.set_keyParams2(sortTileInfo.keyParams2);
    sortTilingData.set_keyParams3(sortTileInfo.keyParams3);
    sortTilingData.set_keyParams4(sortTileInfo.keyParams4);
    sortTilingData.set_keyParams5(sortTileInfo.keyParams5);
    sortTilingData.set_tmpUbSize(sortTileInfo.tmpUbSize);
    sortTilingData.set_lastAxisNum(sortTileInfo.sortAxisNum);
    sortTilingData.set_unsortedDimNum(sortTileInfo.unSortDimNum);
    return;
}

void SetSortTmpSize1(ge::DataType dataType, uint32_t tileData, bool isDescend, SortTileInfo& sortTileInfo)
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
    OP_LOGI("[SortWithIndexTilingForAscendC]", "api of sort shape is %ld, maxUb is %u", realLen, maxValue);
    sortTileInfo.tmpUbSize = maxValue;
    return;
}

uint32_t ComputeRemainUb1(SortTileInfo& sortTileInfo, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint32_t tmpUb = sortTileInfo.ubSize - (ubExtra + tileFactor * tileData);
    OP_LOGD("[SortWithIndexTilingForAscendC]",
            "ComputeRemainUb1 ubSize=%u, ubExtra=%lu, tileFactor=%lu, "
            "tileData=%lu, tmpUb=%lu.",
            sortTileInfo.ubSize, ubExtra, tileFactor, tileData, tmpUb);
    return tmpUb;
}

void AdjTmpUb1(SortTileInfo& sortTileInfo, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint32_t remainUbNew = ComputeRemainUb1(sortTileInfo, tileData, ubExtra, tileFactor) - sortTileInfo.tmpUbSize;
    remainUbNew = remainUbNew > sortTileInfo.blockUbSize ? (remainUbNew - sortTileInfo.blockUbSize) : uint32_t(0);
    uint32_t alignUbSize = (remainUbNew / sortTileInfo.blockUbSize) * sortTileInfo.blockUbSize;
    OP_LOGD("[SortWithIndexTilingForAscendC]",
            "alignUbSize %u, sortTileInfo.tmpUbSize=%lu, "
            "sortTileInfo.blockUbSize=%lu.",
            alignUbSize, sortTileInfo.tmpUbSize, sortTileInfo.blockUbSize);
    sortTileInfo.tmpUbSize = sortTileInfo.tmpUbSize + alignUbSize; // 剩余的ub都给tmpUbsize
}

void ComputeTileDataOne1(SortTileInfo& sortTileInfo, uint32_t lastDimTileNum, uint32_t ubExtra, uint32_t& tileData,
                         uint32_t tileFactor)
{
    uint32_t allCore = CeilDivMul1<uint32_t>(int64_t(lastDimTileNum), int64_t(sortTileInfo.maxCoreNum));
    uint32_t newTileData = CeilDiv1(sortTileInfo.sortAxisNum, int64_t(allCore));
    tileData = CeilDivMul1<uint32_t>(int64_t(newTileData), int64_t(BIN_NUM));
    tileData = std::max(tileData, SMALL_TILE_DATA_NUM);
    SetSortTmpSize1(ge::DT_UINT8, tileData, false, sortTileInfo);
    AdjTmpUb1(sortTileInfo, tileData, ubExtra, tileFactor);
    return;
}

bool NeedAdjTileData1(SortTileInfo& sortTileInfo, uint32_t& tileData, uint32_t lastDimTileNum, uint32_t ubExtra,
                      uint32_t tileFactor)
{
    if (sortTileInfo.unSortDimNum == 1L && lastDimTileNum == 1U) {
        OP_LOGI("[SortWithIndexTilingForAscendC]", "unSortDimNum and lastDimTileNum is 1");
        uint32_t newTileData = CeilDiv1(sortTileInfo.sortAxisNum, int64_t(sortTileInfo.maxCoreNum));
        newTileData = CeilDivMul1<uint32_t>(int64_t(newTileData), int64_t(BIN_NUM));
        tileData = std::max(newTileData, SMALL_TILE_DATA_NUM);
        SetSortTmpSize1(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb1(sortTileInfo, tileData, ubExtra, tileFactor);
        return true;
    }
    if (sortTileInfo.unSortDimNum == 1L || (lastDimTileNum >= sortTileInfo.maxCoreNum)) {
        // b为1时，尽量均匀分核，同时保证处理的最小的tile_data为1024
        OP_LOGI("[SortWithIndexTilingForAscendC]", "unSortDimNum is 1 and lastDimTileNum greater than allCore");
        ComputeTileDataOne1(sortTileInfo, lastDimTileNum, ubExtra, tileData, tileFactor);
        return true;
    }
    if (sortTileInfo.unSortDimNum > 1L && sortTileInfo.unSortDimNum < int64_t(sortTileInfo.maxCoreNum) &&
        lastDimTileNum == 1U) {
        OP_LOGI("[SortWithIndexTilingForAscendC]",
                "unSortDimNum greater than 1,and unSortDimNum small and lastDimTileNum is one");
        uint32_t hCore = sortTileInfo.maxCoreNum / static_cast<uint32_t>(sortTileInfo.unSortDimNum);
        uint32_t hTileData = static_cast<uint32_t>(sortTileInfo.sortAxisNum) / hCore;
        tileData = CeilDivMul1<uint32_t>(int64_t(hTileData), int64_t(BIN_NUM));
        SetSortTmpSize1(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb1(sortTileInfo, tileData, ubExtra, tileFactor);
        return tileData;
    }
    if (sortTileInfo.unSortDimNum > 1L && lastDimTileNum > 1U) {
        // b大于1且h轴循环次数小于总核数，也就是b轴核数大于1
        OP_LOGI("[SortWithIndexTilingForAscendC]", "unSortDimNum is one, lastDimTileNum greater than one");
        int64_t newTileData = sortTileInfo.sortAxisNum / int64_t(lastDimTileNum);
        tileData = CeilDivMul1<uint32_t>(newTileData, int64_t(BIN_NUM));
        lastDimTileNum = CeilDiv1(sortTileInfo.sortAxisNum, int64_t(tileData));
        uint32_t bCore = lastDimTileNum == 0 ? sortTileInfo.maxCoreNum : sortTileInfo.maxCoreNum / lastDimTileNum;
        if (lastDimTileNum < sortTileInfo.maxCoreNum && sortTileInfo.unSortDimNum < int64_t(sortTileInfo.maxCoreNum)) {
            if (sortTileInfo.unSortDimNum < int64_t(bCore)) {
                bCore = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
                uint32_t hCore = sortTileInfo.maxCoreNum / bCore;
                uint32_t tileDataNew = CeilDiv1(int64_t(sortTileInfo.sortAxisNum), int64_t(hCore));
                tileData = CeilDivMul1<uint32_t>(int64_t(tileDataNew), int64_t(BIN_NUM));
            }
        }
        if (bCore == 1U && lastDimTileNum < sortTileInfo.maxCoreNum) {
            ComputeTileDataOne1(sortTileInfo, lastDimTileNum, ubExtra, tileData, tileFactor);
        }
        SetSortTmpSize1(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb1(sortTileInfo, tileData, ubExtra, tileFactor);
        return true;
    }
    return false;
}

uint32_t ComputeTileData1(SortTileInfo& sortTileInfo)
{
    uint32_t ubExtra;
    uint32_t tileFactor;
    if (sortTileInfo.isInt32 == 0U) { // 数据范围超过int32, y2DtypeSize表示索引类型
        ubExtra = UB_CONST_INT64;
        tileFactor = CONST_6 + sortTileInfo.dtypeSize + sortTileInfo.y2DtypeSize;
    } else {
        ubExtra = UB_CONST_INT32;
        tileFactor = CONST_6 + sortTileInfo.dtypeSize + sortTileInfo.y2DtypeSize;
    }

    uint32_t tileData = (sortTileInfo.ubSize - ubExtra) / tileFactor;
    tileData = (tileData / BIN_NUM) * BIN_NUM;
    OP_LOGI("[SortWithIndexTilingForAscendC]",
            "ubExtra=%u, tileFactor=%u, dtypeSize=%u, y2DtypeSize=%lu, "
            "tileData=%lu.",
            ubExtra, tileFactor, sortTileInfo.dtypeSize, sortTileInfo.y2DtypeSize, tileData);
    uint32_t remainUb = ComputeRemainUb1(sortTileInfo, tileData, ubExtra, tileFactor);
    SetSortTmpSize1(ge::DT_UINT8, tileData, false, sortTileInfo);

    uint32_t tmpUbSize = sortTileInfo.tmpUbSize;
    while (tmpUbSize > remainUb) {
        tileData = tileData - BIN_NUM;
        remainUb = ComputeRemainUb1(sortTileInfo, tileData, ubExtra, tileFactor);
        SetSortTmpSize1(ge::DT_UINT8, tileData, false, sortTileInfo);
        tmpUbSize = sortTileInfo.tmpUbSize;
    }
    uint32_t lastDimTileNum = CeilDiv1(sortTileInfo.sortAxisNum, int64_t(tileData));
    OP_LOGI("[SortWithIndexTilingForAscendC]", "tileData %u, lastDimTileNum %u, tmpUbSize %u", tileData, lastDimTileNum,
            tmpUbSize);
    bool smallTile = (sortTileInfo.sortAxisNum <= static_cast<int64_t>(SMALL_TILE_DATA_NUM)) &&
                     lastDimTileNum == uint32_t(1);
    if ((lastDimTileNum % sortTileInfo.maxCoreNum == 0U) || smallTile) {
        OP_LOGI("[SortWithIndexTilingForAscendC]", "lastDimTileNum align or smallTile");
        AdjTmpUb1(sortTileInfo, tileData, ubExtra, tileFactor);
        return tileData;
    }
    if (NeedAdjTileData1(sortTileInfo, tileData, lastDimTileNum, ubExtra, tileFactor)) {
        return tileData;
    }
    AdjTmpUb1(sortTileInfo, tileData, ubExtra, tileFactor);
    return tileData;
}

void ComputeWorkSpace1(gert::TilingContext* context, SortTileInfo& sortTileInfo)
{
    uint32_t dtypeSizeWk = static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.isInt32 == 0U) {
        dtypeSizeWk = static_cast<uint32_t>(sizeof(int64_t));
    }
    size_t excusiveBinsGmWkSize = static_cast<size_t>(sortTileInfo.keyParams1) * sortTileInfo.keyParams4 * dtypeSizeWk;
    excusiveBinsGmWkSize = CeilDivMul1<size_t>(int64_t(excusiveBinsGmWkSize), int64_t(sortTileInfo.blockUbSize));

    size_t globalHistGmWkSize = static_cast<size_t>(sortTileInfo.keyParams3) * sortTileInfo.keyParams2 *
                                sortTileInfo.keyParams0 * dtypeSizeWk;
    globalHistGmWkSize = CeilDivMul1<size_t>(int64_t(globalHistGmWkSize), int64_t(sortTileInfo.blockUbSize));

    size_t outIdxDbWK = static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel *
                        sortTileInfo.y2DtypeSize;
    outIdxDbWK = CeilDivMul1<size_t>(int64_t(outIdxDbWK), int64_t(sortTileInfo.blockUbSize));

    size_t histTileGmWk = static_cast<size_t>(sortTileInfo.lastDimTileNum) * BIN_NUM *
                          sortTileInfo.unsortedDimParallel * sizeof(int16_t) * CONST_2;

    size_t xB8GmWkSize = static_cast<size_t>(sortTileInfo.lastDimTileNum) * sortTileInfo.numTileDataSize *
                         sortTileInfo.unsortedDimParallel;
    xB8GmWkSize = CeilDivMul1<size_t>(int64_t(xB8GmWkSize), int64_t(sortTileInfo.blockUbSize));

    size_t outValueDbWKSize = static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel *
                              sortTileInfo.dtypeSize;
    outValueDbWKSize = CeilDivMul1<size_t>(int64_t(outValueDbWKSize), int64_t(sortTileInfo.blockUbSize));
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    size_t usrSize = excusiveBinsGmWkSize + globalHistGmWkSize + outIdxDbWK + histTileGmWk + xB8GmWkSize +
                     outValueDbWKSize;
    OP_LOGD("[SortWithIndexTilingForAscendC]",
            "excusiveBinsGmWkSize=%lu, globalHistGmWkSize=%lu, histTileGmWk=%lu,"
            " xB8GmWkSize=%lu, outValueDbWKSize=%lu, outIdxDbWK=%lu, usrSize=%lu.",
            excusiveBinsGmWkSize, globalHistGmWkSize, histTileGmWk, xB8GmWkSize, outValueDbWKSize, outIdxDbWK, usrSize);
    userWorkSpaceSize[0] = usrSize + WORK_SPACE_SIZE;
    return;
}

void TileMoreCoreModeOfIdx(gert::TilingContext* context, SortTileInfo& sortTileInfo)
{
    uint32_t tileData = ComputeTileData1(sortTileInfo);
    uint32_t lastDimTileNum = CeilDiv1(int64_t(sortTileInfo.sortAxisNum), int64_t(tileData));
    if (sortTileInfo.maxCoreNum <= lastDimTileNum) {
        sortTileInfo.unsortedDimParallel = 1U;
    } else {
        sortTileInfo.unsortedDimParallel = lastDimTileNum == 0U ? sortTileInfo.maxCoreNum :
                                                                  sortTileInfo.maxCoreNum / lastDimTileNum;
        if (sortTileInfo.unSortDimNum < static_cast<int64_t>(sortTileInfo.unsortedDimParallel)) {
            sortTileInfo.unsortedDimParallel = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
        }
    }
    sortTileInfo.numTileDataSize = tileData;
    sortTileInfo.sortLoopTimes = CeilDiv1(int64_t(sortTileInfo.unSortDimNum),
                                          int64_t(sortTileInfo.unsortedDimParallel));
    sortTileInfo.lastDimNeedCore = std::min(sortTileInfo.maxCoreNum, lastDimTileNum);
    sortTileInfo.coreNumNeed = sortTileInfo.unsortedDimParallel * sortTileInfo.lastDimNeedCore;
    sortTileInfo.lastDimTileNum = lastDimTileNum;

    uint32_t ubSizeNum = sortTileInfo.tmpUbSize / static_cast<uint32_t>(sizeof(uint32_t));
    if (sortTileInfo.isInt32 == 0U) {
        ubSizeNum = sortTileInfo.tmpUbSize / static_cast<uint32_t>(sizeof(int64_t));
    }
    uint32_t allNumGloblHist = BIN_NUM * lastDimTileNum * sortTileInfo.dtypeSize * sortTileInfo.unsortedDimParallel;
    uint32_t allNumExcusiveBin = BIN_NUM * sortTileInfo.dtypeSize * sortTileInfo.unsortedDimParallel;
    uint32_t oneCoreSize = CeilDiv1(int64_t(allNumGloblHist), int64_t(sortTileInfo.coreNumNeed));
    sortTileInfo.keyParams5 = std::max(static_cast<int64_t>(oneCoreSize),
                                       static_cast<int64_t>(sortTileInfo.blockUbSize));
    sortTileInfo.keyParams0 = CeilDiv1(int64_t(allNumGloblHist), int64_t(sortTileInfo.keyParams5));
    sortTileInfo.keyParams3 = CeilDiv1(int64_t(sortTileInfo.keyParams5), int64_t(ubSizeNum));
    sortTileInfo.keyParams2 = sortTileInfo.keyParams5 > ubSizeNum ? ubSizeNum : sortTileInfo.keyParams5;

    uint32_t oneCoreSize1 = CeilDiv1(int64_t(allNumExcusiveBin), int64_t(sortTileInfo.coreNumNeed));
    sortTileInfo.keyParams4 = std::max(static_cast<int64_t>(oneCoreSize1),
                                       static_cast<int64_t>(sortTileInfo.blockUbSize));

    sortTileInfo.keyParams1 = CeilDiv1(int64_t(allNumExcusiveBin), int64_t(sortTileInfo.keyParams4));
    ComputeWorkSpace1(context, sortTileInfo);
    context->SetScheduleMode(1);
    return;
}

ge::graphStatus RadixSortTilingOfIdx(gert::TilingContext* context, int32_t maxCoreNum)
{
    OP_LOGI(context->GetNodeName(), "SortWithIndexTIling start");
    SortWithIndexTilingDataSimt sortTilingData;
    const gert::Shape inputShape = Ops::Base::EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    auto dataType = context->GetInputDesc(0)->GetDataType();
    // out index type
    auto y2DType = context->GetOutputDesc(1)->GetDataType();

    OP_CHECK_IF(
        tilingDataTypeKeyMap.count(dataType) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                  "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16, BF16"),
        return ge::GRAPH_FAILED);
    auto tilingKey = tilingDataTypeKeyMap.find(dataType)->second;
    std::string opType(context->GetNodeType());

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(SIMT_UB),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                                      "The value of ubSize must be greater than SIMT_UB."),
                return ge::GRAPH_FAILED);

    OP_LOGW(context->GetNodeName(), "Get op_type[%s]", opType.c_str());

    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    uint32_t xDimNum = inputShape.GetDimNum();
    const bool* isDescending = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, isDescending);
    OP_LOGI(context->GetNodeName(), "isDescending=%u", *isDescending);
    int64_t sortAxisNum = inputShape.GetDim(xDimNum - 1);
    uint64_t unSortDimNum = 1;
    for (uint32_t i = 0u; i < static_cast<uint32_t>(xDimNum - 1); i++) {
        unSortDimNum *= inputShape.GetDim(i);
    }

    uint32_t isInInt32Range = static_cast<uint32_t>(sortAxisNum <= INT32_MAX_RANGE_VALUE);
    sortTilingData.set_isInInt32Range(isInInt32Range);

    uint32_t tileData = TILE_DATA_NUM;
    if (dataType == ge::DT_UINT64 || dataType == ge::DT_INT64) {
        tileData = TILE_DATA_NUM_B64;
    } else {
        tileData = TILE_DATA_NUM;
    }

    SortTileInfo sortTileInfo;
    // 预留给SIMT使用
    sortTileInfo.ubSize = ubSize - SIMT_UB;
    uint32_t blockUbAglinSize = Ops::Base::GetUbBlockSize(context);
    sortTileInfo.blockUbSize = blockUbAglinSize;
    sortTileInfo.dtypeSize = tilingDataTypeBitMap.find(dataType)->second;
    sortTileInfo.y2DtypeSize = tilingDataTypeBitMap.find(y2DType)->second;
    sortTileInfo.maxCoreNum = maxCoreNum;
    sortTileInfo.dataType = dataType;
    sortTileInfo.xDimNum = xDimNum;
    sortTileInfo.sortAxisNum = sortAxisNum;
    sortTileInfo.unSortDimNum = unSortDimNum;
    sortTileInfo.isInt32 = isInInt32Range;
    sortTileInfo.isDescend = isDescending;
    sortTileInfo.numTileDataSize = tileData;

    // 设置高级api tmpUbSize需要的空间
    SetSortTmpSizeOfIdx(dataType, sortAxisNum, tileData, *isDescending, true, sortTilingData, sortTileInfo);
    if (sortAxisNum <= SMALL_SORT_MAX_DATA_SIZE && optDataTypeBitMap.count(dataType) != 0) {
        sortTilingData.set_modeType(SMALL_SIZE_OPTIM_MODE);
        uint32_t tileDataS = TILE_DATA_NUM;
        TileModeSmallSizeOptimOfIdx(unSortDimNum, maxCoreNum, sortAxisNum, tileDataS, sortTileInfo);
        SetMergeSortTmpSizeOfIdx(context, dataType, sortAxisNum, sortTilingData);
        tilingKey += MERGE_SORT_TILING_OFFSET;
    } else if (sortAxisNum <= static_cast<int64_t>(tileData)) {
        sortTilingData.set_modeType(SMALL_SIZE_MODE);
        TileModeSmallSizeOfIdx(context, unSortDimNum, maxCoreNum, sortAxisNum, sortTileInfo);
    } else {
        // more core radix sort case
        sortTilingData.set_modeType(MULT_CORE_MODE);
        TileMoreCoreModeOfIdx(context, sortTileInfo);
    }
    OP_LOGI(context->GetNodeName(),
            "ubSize: %ld, ubAglinSize: %ld, dtypeSize: %u, y2DtypeSize=%u,"
            " sortTileInfo.ubSize=%u, maxCoreNum=%lu.",
            ubSize, blockUbAglinSize, sortTileInfo.dtypeSize, sortTileInfo.y2DtypeSize, sortTileInfo.ubSize,
            maxCoreNum);

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(sortTileInfo.coreNumNeed);
    context->SetScheduleMode(1);
    sortTilingData.set_isDescend(*isDescending);
    sortTilingData.set_lastAxisNum(sortAxisNum);
    sortTilingData.set_unsortedDimNum(unSortDimNum);
    sortTilingData.set_oneCoreRowNum(sortTileInfo.oneCoreRowNum);
    sortTilingData.set_outputLastDimValue(sortAxisNum);
    FillRadixSortTilingDataSort(sortTileInfo, sortTilingData);
    sortTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(sortTilingData.GetDataSize());
    context->SetLocalMemorySize(sortTileInfo.ubSize);
    PrintTilingDataOfIdx(sortTileInfo, sortTilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortWithIndexTilingSimt(gert::TilingContext* context, int32_t maxCoreNum)
{
    return RadixSortTilingOfIdx(context, maxCoreNum);
}

static ge::graphStatus Tiling4SortWithIndex(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const SortWithIndexCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    OP_LOGD(context->GetNodeName(), "AscendC SortWithIndex simt tiling");
    OP_CHECK_IF(SortWithIndexTilingSimt(context, compile_info->core_num) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SortWithIndexTilingSimt", "GRAPH_FAILED",
                                                      "The value of SortWithIndexTilingSimt must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4SortWithIndex(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "AscendC Tiling starting GRAPH_SUCCESS");
    auto compileInfo = context->GetCompiledInfo<SortWithIndexCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->core_num <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum",
                                                      std::to_string(compileInfo->core_num).c_str(),
                                                      "The value of coreNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SortWithIndex)
    .Tiling(Tiling4SortWithIndex)
    .TilingParse<SortWithIndexCompileInfo>(TilingPrepare4SortWithIndex);
} // namespace optiling
