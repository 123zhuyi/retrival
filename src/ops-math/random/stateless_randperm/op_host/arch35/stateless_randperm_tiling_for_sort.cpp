/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file stateless_randperm_tiling_for_sort.cpp
 * \brief sort tiling for randperm impl，可搜索"侵入修改"
 */
#include <iostream>
#include <string>

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/platform_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "stateless_randperm_tiling_for_sort.h"

namespace optiling {
namespace statelessRandpermTiling{
constexpr size_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
const uint32_t BIN_NUM = 256;                   // 直方图一次处理256B
const uint32_t SMALL_TILE_DATA_NUM = 1024;      // 测试数据得出一次至少处理1024，sort性能比较好
const uint32_t SIMT_UB = 32768;                 // 预留了32k给simt使用
const uint32_t SMALL_SORT_MAX_DATA_SIZE = 1024; // 走merge sort条件
const uint32_t TILE_DATA_NUM = 4096;            // merge_sort一次ub处理的数据量
const uint32_t UB_CONST_INT32 = 4096;           // 输出idx为int32时kernel侧需要的固定ub大小
const uint32_t UB_CONST_INT64 = 7168;           // 输出idx为int64时kernel侧需要的固定ub大小
const uint32_t CONST_10 = 10;
const uint32_t CONST_14 = 14;
const uint64_t SCH_ID_2 = 2;
const uint64_t SCH_ID_3 = 3;
const uint32_t CONST_2 = 2;
const uint32_t MULTI_CORE_MERGE_SORT_MAX_SIZE = 32768;
const uint32_t MERGE_SORT_DEALING_LIST_NUM = 4;
const uint32_t MERGE_SORT_DATASIZE = 8;
const int64_t MERGE_SORT_WORKSPACE_PARAM = 5;
const int64_t ONE_CORE_DATA_SIZE = 1024;

struct SortTileInfo {
    uint32_t coreNumNeed = 0;
    uint32_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
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
static const std::map<ge::DataType, uint32_t> tilingDataTypeBitMap = {
    { ge::DT_INT64, 8 },  { ge::DT_INT32, 4 },   { ge::DT_INT16, 2 },  { ge::DT_INT8, 1 },
    { ge::DT_UINT64, 8 }, { ge::DT_UINT32, 4 },  { ge::DT_UINT16, 2 }, { ge::DT_UINT8, 1 },
    { ge::DT_FLOAT, 4 },  { ge::DT_FLOAT16, 2 }, { ge::DT_BF16, 2 }
};
static const std::map<ge::DataType, uint32_t> mergeType = { { ge::DT_FLOAT, 4 },
    { ge::DT_FLOAT16, 2 },
    { ge::DT_BF16, 2 } };

uint32_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return static_cast<uint32_t>(a);
    }
    return static_cast<uint32_t>((a + b - 1) / b);
}

template <typename T>
auto CeilDivMul(int64_t a, int64_t b) ->T const
{
    if (b == 0) {
        return static_cast<T>(a);
    }
    return static_cast<T>(((a + b - 1) / b) * b);
}

ge::graphStatus CheckInputAndOutput(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo); 
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize <= static_cast<uint64_t>(SIMT_UB)) {
        std::string valueStr = std::to_string(ubSize);
        std::string reasonMsg = "allUb must be greater than simtUb";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ub size", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.blockUbSize = Ops::Base::GetUbBlockSize(context);
    sortTileInfo.ubSize = ubSize - SIMT_UB;       // 侵入修改：MODIFY
    OP_LOGI(context->GetNodeName(), "ubSize is %ld, simtDcache is %u, blockUbSize %u", sortTileInfo.ubSize, SIMT_UB, sortTileInfo.blockUbSize);  // 侵入修改：MODIFY
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape &inputShape = Ops::Base::EnsureNotScalar(inputShapePtr->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    const gert::Shape &outShape = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());
    auto yStorage1 = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage1);
    const gert::Shape &outShape1 = Ops::Base::EnsureNotScalar(yStorage1->GetStorageShape());
    if (inputShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0) {
        std::string valueStr = std::to_string(inputShape.GetShapeSize()) + " and " + std::to_string(outShape.GetShapeSize());
        std::string reasonMsg = "not support empty input or output";
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context->GetNodeName(), "input and output", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (outShape != outShape1 || outShape != inputShape) {
        std::string valueStr = Ops::Base::ToString(inputShape) + " and " + Ops::Base::ToString(outShape);
        std::string reasonMsg = "input and outputs shape must be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "input and output", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
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

ge::graphStatus SortCheckParams(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    OP_CHECK_IF(CheckInputAndOutput(context, sortTileInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CheckInputAndOutput failed"), return ge::GRAPH_FAILED);
    auto inputDescPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
    ge::DataType dataType = inputDescPtr->GetDataType();
    if (tilingDataTypeBitMap.count(dataType) == 0) {
        std::string valueStr = Ops::Base::ToString(dataType);
        std::string reasonMsg = "Not supported data type";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "input tensor x", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.dataType = dataType;
    sortTileInfo.dtypeSize = tilingDataTypeBitMap.find(dataType)->second;
    auto outDescPtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr);
    auto y2DType = outDescPtr->GetDataType();
    auto outDescPtr0 = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr0);
    auto y1DType = outDescPtr0->GetDataType();
    if ((y2DType != ge::DT_INT64) && (y2DType != ge::DT_INT32)) {
        std::string valueStr = Ops::Base::ToString(y2DType);
        std::string reasonMsg = "y2 dtype only support int64 or int32";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "output tensor y2", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (y1DType != dataType) {
        std::string valueStr = Ops::Base::ToString(dataType) + " and " + Ops::Base::ToString(y1DType);
        std::string reasonMsg = "input0 dtype must be same as output0 dtype";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(), "input and output", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.y2DtypeSize = tilingDataTypeBitMap.find(y2DType)->second;
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *isDescending = attrs->GetAttrPointer<bool>(1);
    const int64_t *sortAxisPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, isDescending);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortAxisPtr);
    int32_t sortAxis = static_cast<int32_t>(*sortAxisPtr);
    sortAxis = sortAxis < 0 ? (sortAxis + sortTileInfo.xDimNum) : sortAxis;
    if (sortAxis != (sortTileInfo.xDimNum - 1)) {
        std::string valueStr = std::to_string(sortAxis);
        std::string reasonMsg = "only support last dim sort";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr sort_axis", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SetSortTmpSize(ge::DataType dataType, uint32_t tileData, bool isDescend, SortTileInfo &sortTileInfo)
{
    int64_t realLen = std::min(sortTileInfo.sortAxisNum, static_cast<int64_t>(tileData));
    std::vector<int64_t> shapeVec = { realLen };
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

bool IsMergeSort(SortTileInfo &sortTileInfo)
{
    bool support =
        (sortTileInfo.sortAxisNum <= SMALL_SORT_MAX_DATA_SIZE) && (mergeType.count(sortTileInfo.dataType) != 0);
    return support;
}

bool IsMergeSortMultiCore(SortTileInfo &sortTileInfo)
{
    bool isMuiltiCoreMergeSort =
        ((sortTileInfo.unSortDimNum == 1) && (sortTileInfo.sortAxisNum <= MULTI_CORE_MERGE_SORT_MAX_SIZE) &&
         (sortTileInfo.sortAxisNum > SMALL_SORT_MAX_DATA_SIZE) && (sortTileInfo.dataType == ge::DT_FLOAT));
    return isMuiltiCoreMergeSort;
}

bool IsRadixSortOneCore(SortTileInfo &sortTileInfo)
{
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        return false;
    }
    uint32_t xUbSize = sortTileInfo.sortAxisNum * sortTileInfo.dtypeSize;
    xUbSize = CeilDivMul<uint32_t>(int64_t(xUbSize), int64_t(sortTileInfo.blockUbSize));
    uint32_t y2UbSize = sortTileInfo.sortAxisNum * static_cast<uint32_t>(sizeof(int32_t));
    y2UbSize = CeilDivMul<uint32_t>(int64_t(y2UbSize), int64_t(sortTileInfo.blockUbSize));
    uint32_t halfNum = y2UbSize / static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.y2DtypeSize == static_cast<uint32_t>(sizeof(int64_t))) {
        y2UbSize = y2UbSize * CONST_2;
    }
    int64_t remainUb = static_cast<int64_t>(sortTileInfo.ubSize) - static_cast<int64_t>(xUbSize * CONST_2 + y2UbSize);
    if (remainUb <= static_cast<int64_t>(0)) {
        return false;
    }
    sortTileInfo.keyParams0 = xUbSize;
    sortTileInfo.keyParams1 = y2UbSize;
    sortTileInfo.keyParams2 = halfNum;
    remainUb = (remainUb / int64_t(sortTileInfo.blockUbSize)) * int64_t(sortTileInfo.blockUbSize);
    SetSortTmpSize(sortTileInfo.dataType, sortTileInfo.sortAxisNum, sortTileInfo.isDescend, sortTileInfo);
    int64_t tmpUb = static_cast<int64_t>(sortTileInfo.tmpUbSize);
    OP_LOGI("RadixSortTiling", "remainUb is %ld, tmpUb is %ld", remainUb, tmpUb);
    return tmpUb <= remainUb;
}

uint32_t ComputeRemainUb(SortTileInfo &sortTileInfo, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint32_t tmpUb = sortTileInfo.ubSize - (ubExtra + tileFactor * tileData);
    return tmpUb;
}

void AdjTmpUb(SortTileInfo &sortTileInfo, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint32_t remainUbNew = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor) - sortTileInfo.tmpUbSize;
    remainUbNew = remainUbNew > sortTileInfo.blockUbSize ? (remainUbNew - sortTileInfo.blockUbSize) : uint32_t(0);
    uint32_t alignUbSize = (remainUbNew / sortTileInfo.blockUbSize) * sortTileInfo.blockUbSize;
    OP_LOGI("RadixSortTiling", "alignUbSize %u", alignUbSize);
    sortTileInfo.tmpUbSize = sortTileInfo.tmpUbSize + alignUbSize; // 剩余的ub都给tmpUbsize
}

void ComputeTileDataOne(SortTileInfo &sortTileInfo, uint32_t lastDimTileNum,  uint32_t ubExtra, uint32_t &tileData,
                        uint32_t tileFactor)
{
    uint32_t allCore = CeilDivMul<uint32_t>(int64_t(lastDimTileNum), int64_t(sortTileInfo.maxCoreNum));
    uint32_t newTileData = CeilDiv(sortTileInfo.sortAxisNum, int64_t(allCore));
    tileData = CeilDivMul<uint32_t>(int64_t(newTileData), int64_t(BIN_NUM));
    tileData = std::max(tileData, SMALL_TILE_DATA_NUM);
    SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
    AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
    return;
}

bool NeedAdjTileData(SortTileInfo &sortTileInfo, uint32_t &tileData, uint32_t lastDimTileNum, uint32_t ubExtra,
                     uint32_t tileFactor)
{
    if (sortTileInfo.unSortDimNum == int64_t(1) && lastDimTileNum == uint32_t(1)) {
        OP_LOGI("RadixSortTiling", "unSortDimNum and lastDimTileNum is 1");
        uint32_t newTileData = CeilDiv(sortTileInfo.sortAxisNum, int64_t(sortTileInfo.maxCoreNum));
        newTileData = CeilDivMul<uint32_t>(int64_t(newTileData), int64_t(BIN_NUM));
        tileData = std::max(newTileData, SMALL_TILE_DATA_NUM);
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
        lastDimTileNum == uint32_t(1)) {
        OP_LOGI("RadixSortTiling", "unSortDimNum greater than 1,and unSortDimNum small and lastDimTileNum is one");
        uint32_t hCore = sortTileInfo.maxCoreNum / static_cast<uint32_t>(sortTileInfo.unSortDimNum);
        uint32_t hTileData = static_cast<uint32_t>(sortTileInfo.sortAxisNum) / hCore;
        tileData = CeilDivMul<uint32_t>(int64_t(hTileData), int64_t(BIN_NUM));
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        AdjTmpUb(sortTileInfo, tileData, ubExtra, tileFactor);
        return true;
    }
    if (sortTileInfo.unSortDimNum > int64_t(1) && lastDimTileNum > uint32_t(1)) {
        // b大于1且h轴循环次数小于总核数，也就是b轴核数大于1
        OP_LOGI("RadixSortTiling", "unSortDimNum is one, lastDimTileNum greater than one");
        int64_t newTileData = sortTileInfo.sortAxisNum / int64_t(lastDimTileNum);
        tileData = CeilDivMul<uint32_t>(newTileData, int64_t(BIN_NUM));
        lastDimTileNum = CeilDiv(sortTileInfo.sortAxisNum, int64_t(tileData));
        uint32_t bCore = lastDimTileNum == 0 ? sortTileInfo.maxCoreNum : sortTileInfo.maxCoreNum / lastDimTileNum;
        if (lastDimTileNum < sortTileInfo.maxCoreNum && sortTileInfo.unSortDimNum < int64_t(sortTileInfo.maxCoreNum)) {
            if (sortTileInfo.unSortDimNum < int64_t(bCore)) {
                bCore = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
                uint32_t hCore = sortTileInfo.maxCoreNum / bCore;
                uint32_t tileDataNew = CeilDiv(int64_t(sortTileInfo.sortAxisNum), int64_t(hCore));
                tileData = CeilDivMul<uint32_t>(int64_t(tileDataNew), int64_t(BIN_NUM));
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

uint32_t ComputeTileData(SortTileInfo &sortTileInfo)
{
    uint32_t ubExtra;
    uint32_t tileFactor;
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        ubExtra = UB_CONST_INT64;
        tileFactor = CONST_14 + sortTileInfo.dtypeSize;
    } else {
        ubExtra = UB_CONST_INT32;
        tileFactor = CONST_10 + sortTileInfo.dtypeSize;
    }

    uint32_t tileData = (sortTileInfo.ubSize - ubExtra) / tileFactor;
    tileData = (tileData / BIN_NUM) * BIN_NUM;

    uint32_t remainUb = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor);
    SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);

    uint32_t tmpUbSize = sortTileInfo.tmpUbSize;
    while (tmpUbSize > remainUb) {
        tileData = tileData - BIN_NUM;
        remainUb = ComputeRemainUb(sortTileInfo, tileData, ubExtra, tileFactor);
        SetSortTmpSize(ge::DT_UINT8, tileData, false, sortTileInfo);
        tmpUbSize = sortTileInfo.tmpUbSize;
    }
    uint32_t lastDimTileNum = CeilDiv(sortTileInfo.sortAxisNum, int64_t(tileData));
    OP_LOGI("RadixSortTiling", "tileData %u, lastDimTileNum %u, tmpUbSize %u", tileData, lastDimTileNum, tmpUbSize);
    bool smallTile =
        (sortTileInfo.sortAxisNum <= static_cast<int64_t>(SMALL_TILE_DATA_NUM)) && lastDimTileNum == uint32_t(1);
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

void GetMergeSortMultiCore(gert::TilingContext *context, SortTileInfo &sortTileInfo) {
    uint32_t coreNumNeed = static_cast<uint32_t>((sortTileInfo.sortAxisNum + ONE_CORE_DATA_SIZE - 1) / ONE_CORE_DATA_SIZE);
    uint32_t tileNum = static_cast<uint32_t>(sortTileInfo.sortAxisNum) / coreNumNeed;
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    sortTileInfo.lastDimNeedCore = coreNumNeed;
    sortTileInfo.numTileDataSize = tileNum;
	sortTileInfo.coreNumNeed = coreNumNeed;

    uint32_t byteNum = MERGE_SORT_DEALING_LIST_NUM * MERGE_SORT_DATASIZE * 2;//4list 8byte 2input/output
    byteNum += MERGE_SORT_DEALING_LIST_NUM * static_cast<uint32_t>(sizeof(uint32_t));//extract index
    if (sortTileInfo.y2DtypeSize == sizeof(int64_t)) {
        byteNum += MERGE_SORT_DEALING_LIST_NUM * static_cast<uint32_t>(sizeof(int64_t));
    }
    if (sortTileInfo.dataType == ge::DT_BF16) {
        byteNum += MERGE_SORT_DEALING_LIST_NUM * mergeType.find(ge::DT_BF16)->second;
		byteNum += MERGE_SORT_DEALING_LIST_NUM * static_cast<uint32_t>(sizeof(float));//extract value
    } else {
		byteNum += MERGE_SORT_DEALING_LIST_NUM * tilingDataTypeBitMap.find(sortTileInfo.dataType)->second;//extract value
	}
    sortTileInfo.keyParams0 = sortTileInfo.ubSize / byteNum;

    OP_LOGI("[mergeSort]", "maxDealingNum: %u", sortTileInfo.keyParams0);
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    size_t usrSize = static_cast<size_t>(
        MERGE_SORT_WORKSPACE_PARAM * sortTileInfo.sortAxisNum * static_cast<uint32_t>(sizeof(int32_t)));
    userWorkSpaceSize[0] = usrSize + WORK_SPACE_SIZE;
    context->SetScheduleMode(1);
    return;
}

void GetRadixSortOneCore(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    sortTileInfo.lastDimNeedCore = static_cast<uint32_t>(1);
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    sortTileInfo.sortLoopTimes = CeilDiv(sortTileInfo.unSortDimNum, int64_t(sortTileInfo.maxCoreNum));
    if (sortTileInfo.sortLoopTimes > static_cast<uint32_t>(1)) {
        sortTileInfo.coreNumNeed = sortTileInfo.maxCoreNum;
    } else {
        uint32_t core = static_cast<uint32_t>(sortTileInfo.unSortDimNum) % sortTileInfo.maxCoreNum;
        sortTileInfo.coreNumNeed = core == uint32_t(0) ? sortTileInfo.maxCoreNum : core;
    }
    sortTileInfo.unsortedDimParallel = sortTileInfo.coreNumNeed;
    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    return;
}

void ComputeWorkSpace(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    uint32_t dtypeSizeWk = static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        dtypeSizeWk = static_cast<uint32_t>(sizeof(int64_t));
    }
    size_t excusiveBinsGmWkSize = static_cast<size_t>(sortTileInfo.keyParams1) * sortTileInfo.keyParams4 * dtypeSizeWk;
    excusiveBinsGmWkSize = CeilDivMul<size_t>(int64_t(excusiveBinsGmWkSize), int64_t(sortTileInfo.blockUbSize));

    size_t globalHistGmWkSize =
        static_cast<size_t>(sortTileInfo.keyParams3) * sortTileInfo.keyParams2 * sortTileInfo.keyParams0 * dtypeSizeWk;
    globalHistGmWkSize = CeilDivMul<size_t>(int64_t(globalHistGmWkSize), int64_t(sortTileInfo.blockUbSize));

    size_t outIdxDbWK = static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel * dtypeSizeWk;
    outIdxDbWK = CeilDivMul<size_t>(int64_t(outIdxDbWK), int64_t(sortTileInfo.blockUbSize));

    size_t histTileGmWk = static_cast<size_t>(sortTileInfo.lastDimTileNum) * BIN_NUM * sortTileInfo.unsortedDimParallel *
        sizeof(int16_t) * CONST_2;

    size_t xB8GmWkSize = static_cast<size_t>(sortTileInfo.lastDimTileNum) * sortTileInfo.numTileDataSize *
        sortTileInfo.unsortedDimParallel;
    xB8GmWkSize = CeilDivMul<size_t>(int64_t(xB8GmWkSize), int64_t(sortTileInfo.blockUbSize));

    size_t outValueDbWKSize =
        static_cast<size_t>(sortTileInfo.sortAxisNum) * sortTileInfo.unsortedDimParallel * sortTileInfo.dtypeSize;
    outValueDbWKSize = CeilDivMul<size_t>(int64_t(outValueDbWKSize), int64_t(sortTileInfo.blockUbSize));

    OP_LOGI("RadixSortTiling",
        "excusiveBinsGmWkSize %lu, globalHistGmWkSize %lu, outIdxDbWK %lu, histTileGmWk %lu,"
        " xB8GmWkSize %lu, outValueDbWKSize %lu ",
        excusiveBinsGmWkSize, globalHistGmWkSize, outIdxDbWK, histTileGmWk, xB8GmWkSize, outValueDbWKSize);
    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    size_t usrSize =
        excusiveBinsGmWkSize + globalHistGmWkSize + outIdxDbWK + histTileGmWk + xB8GmWkSize + outValueDbWKSize;
    userWorkSpaceSize[0] = usrSize + WORK_SPACE_SIZE;
    return;
}

void GetRadixSortMoreCore(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    // 侵入修改：DELETE。在stateless_randperm里已经减去SIMT的空间，这里不需要再减
    uint32_t tileData = ComputeTileData(sortTileInfo);
    uint32_t lastDimTileNum = CeilDiv(int64_t(sortTileInfo.sortAxisNum), int64_t(tileData));
    if (sortTileInfo.maxCoreNum <= lastDimTileNum) {
        sortTileInfo.unsortedDimParallel = static_cast<uint32_t>(1);
    } else {
        sortTileInfo.unsortedDimParallel = lastDimTileNum == 0 ? sortTileInfo.maxCoreNum : sortTileInfo.maxCoreNum / lastDimTileNum;
        if (sortTileInfo.unSortDimNum < static_cast<int64_t>(sortTileInfo.unsortedDimParallel)) {
            sortTileInfo.unsortedDimParallel = static_cast<uint32_t>(sortTileInfo.unSortDimNum);
        }
    }
    sortTileInfo.numTileDataSize = tileData;
    sortTileInfo.sortLoopTimes = CeilDiv(int64_t(sortTileInfo.unSortDimNum), int64_t(sortTileInfo.unsortedDimParallel));
    sortTileInfo.lastDimNeedCore = std::min(sortTileInfo.maxCoreNum, lastDimTileNum);
    sortTileInfo.coreNumNeed = sortTileInfo.unsortedDimParallel * sortTileInfo.lastDimNeedCore;
    sortTileInfo.lastDimTileNum = lastDimTileNum;

    uint32_t ubSizeNum = sortTileInfo.tmpUbSize / static_cast<uint32_t>(sizeof(uint32_t));
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        ubSizeNum = sortTileInfo.tmpUbSize / static_cast<uint32_t>(sizeof(int64_t));
    }
    uint32_t allNumGloblHist = BIN_NUM * lastDimTileNum * sortTileInfo.dtypeSize * sortTileInfo.unsortedDimParallel;
    uint32_t allNumExcusiveBin = BIN_NUM * sortTileInfo.dtypeSize * sortTileInfo.unsortedDimParallel;
    uint32_t oneCoreSize = CeilDiv(int64_t(allNumGloblHist), int64_t(sortTileInfo.coreNumNeed));
    sortTileInfo.keyParams5 =
        std::max(static_cast<int64_t>(oneCoreSize), static_cast<int64_t>(sortTileInfo.blockUbSize));
    sortTileInfo.keyParams0 = CeilDiv(int64_t(allNumGloblHist), int64_t(sortTileInfo.keyParams5));
    sortTileInfo.keyParams3 = CeilDiv(int64_t(sortTileInfo.keyParams5), int64_t(ubSizeNum));
    sortTileInfo.keyParams2 = sortTileInfo.keyParams5 > ubSizeNum ? ubSizeNum : sortTileInfo.keyParams5;

    uint32_t oneCoreSize1 = CeilDiv(int64_t(allNumExcusiveBin), int64_t(sortTileInfo.coreNumNeed));
    sortTileInfo.keyParams4 =
        std::max(static_cast<int64_t>(oneCoreSize1), static_cast<int64_t>(sortTileInfo.blockUbSize));

    sortTileInfo.keyParams1 = CeilDiv(int64_t(allNumExcusiveBin), int64_t(sortTileInfo.keyParams4));
    ComputeWorkSpace(context, sortTileInfo);
    context->SetScheduleMode(1);
    return;
}

void FillTilingDataSort(SortTileInfo &sortTileInfo, SortRegBaseTilingData *sortTilingData)
{
    sortTilingData->numTileDataSize = sortTileInfo.numTileDataSize;
    sortTilingData->unsortedDimParallel = sortTileInfo.unsortedDimParallel;
    sortTilingData->lastDimTileNum = sortTileInfo.lastDimTileNum;
    sortTilingData->sortLoopTimes = sortTileInfo.sortLoopTimes;
    sortTilingData->lastDimNeedCore = sortTileInfo.lastDimNeedCore;
    sortTilingData->keyParams0 = sortTileInfo.keyParams0;
    sortTilingData->keyParams1 = sortTileInfo.keyParams1;
    sortTilingData->keyParams2 = sortTileInfo.keyParams2;
    sortTilingData->keyParams3 = sortTileInfo.keyParams3;
    sortTilingData->keyParams4 = sortTileInfo.keyParams4;
    sortTilingData->keyParams5 = sortTileInfo.keyParams5;
    sortTilingData->tmpUbSize = sortTileInfo.tmpUbSize;
    sortTilingData->lastAxisNum = sortTileInfo.sortAxisNum;
    sortTilingData->unsortedDimNum = sortTileInfo.unSortDimNum;
    return;
}

void PrintTilindDataSort(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    OP_LOGI(context->GetNodeName(),
        "realCoreNum %u, numTileDataSize %u, unsortedDimParallel %u, "
        "lastDimTileNum %u, sortLoopTimes %u, lastDimNeedCore %u, keyParams0 %u, keyParams1 %u "
        "keyParams2 %u, keyParams3 %u, keyParams4 %u, keyParams5 %u, tmpUbSize %u, "
        "lastAxisNum %ld, unsortedDimNum %ld ",
        sortTileInfo.coreNumNeed, sortTileInfo.numTileDataSize, sortTileInfo.unsortedDimParallel,
        sortTileInfo.lastDimTileNum, sortTileInfo.sortLoopTimes, sortTileInfo.lastDimNeedCore, sortTileInfo.keyParams0,
        sortTileInfo.keyParams1, sortTileInfo.keyParams2, sortTileInfo.keyParams3, sortTileInfo.keyParams4,
        sortTileInfo.keyParams5, sortTileInfo.tmpUbSize, sortTileInfo.sortAxisNum, sortTileInfo.unSortDimNum);
    return;
}

void GetMergeSort(gert::TilingContext *context, SortTileInfo &sortTileInfo)
{
    uint32_t alignNum = CeilDivMul<uint32_t>(int64_t(sortTileInfo.sortAxisNum), int64_t(sortTileInfo.blockUbSize));
    if (alignNum == 0) {
        OP_LOGE(context->GetNodeName(), "alignNum is 0");
        return;
    }
    uint32_t tileData = TILE_DATA_NUM;
    uint32_t oneCoreRowNum = (tileData / CONST_2) / alignNum;
    oneCoreRowNum = (oneCoreRowNum == uint32_t(0) ? uint32_t(1) : oneCoreRowNum);
    uint32_t virUnsortedDimNum = CeilDiv(sortTileInfo.unSortDimNum, int64_t(oneCoreRowNum));
    uint32_t coreNumNeed = 0;
    uint32_t sortLoopTimes = CeilDiv(int64_t(virUnsortedDimNum), int64_t(sortTileInfo.maxCoreNum));
    if (sortLoopTimes == static_cast<uint32_t>(1)) {
        uint32_t realCoreNum = virUnsortedDimNum % sortTileInfo.maxCoreNum;
        if (realCoreNum == static_cast<uint32_t>(0)) {
            realCoreNum = sortTileInfo.maxCoreNum;
        }
        coreNumNeed = realCoreNum;
    } else {
        coreNumNeed = sortTileInfo.maxCoreNum;
    }
    uint32_t maxTypeSize =
        (sortTileInfo.dataType == ge::DT_BF16) ? mergeType.find(ge::DT_FLOAT)->second : sortTileInfo.dtypeSize;
    auto platform_info = context->GetPlatformInfo();
    auto plat = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t dataSizeNeed = AscendC::GetConcatTmpSize(plat, alignNum, maxTypeSize);
    dataSizeNeed = std::max(dataSizeNeed, sortTileInfo.blockUbSize);
    sortTileInfo.tmpUbSize = dataSizeNeed;
    sortTileInfo.sortLoopTimes = sortLoopTimes;
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    sortTileInfo.unsortedDimParallel = coreNumNeed;
    sortTileInfo.lastDimNeedCore = static_cast<uint32_t>(1);
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.sortAxisNum);
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.keyParams0 = oneCoreRowNum;
    sortTileInfo.keyParams1 = alignNum * oneCoreRowNum * sortTileInfo.dtypeSize;
    sortTileInfo.keyParams2 = alignNum * oneCoreRowNum * sortTileInfo.y2DtypeSize;
    sortTileInfo.keyParams3 = alignNum;
    size_t *userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
}

ge::graphStatus RadixSortTiling(gert::TilingContext *context, int32_t maxCoreNum)
{
    SortRegBaseTilingData *sortTilingData{ nullptr };
    sortTilingData = context->GetTilingData<SortRegBaseTilingData>();
    OP_CHECK_IF(sortTilingData == nullptr,
        OP_LOGE(context->GetNodeName(), "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(sortTilingData, sizeof(SortRegBaseTilingData), 0, sizeof(SortRegBaseTilingData)) != EOK),
        OP_LOGE(context->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    SortTileInfo sortTileInfo;
    OP_CHECK_IF(SortCheckParams(context, sortTileInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check params failed"), return ge::GRAPH_FAILED);
    sortTileInfo.maxCoreNum = static_cast<uint32_t>(maxCoreNum);
    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    uint64_t isInt32 = static_cast<uint64_t>((sortTileInfo.sortAxisNum <= int32Max));
    const bool *isDescending = context->GetAttrs()->GetAttrPointer<bool>(1);
    uint64_t isDescend = *isDescending;
    sortTileInfo.isDescend = static_cast<bool>(isDescend);
    sortTileInfo.isInt32 = static_cast<uint32_t>(isInt32);
    OP_LOGI(context->GetNodeName(), "isInt32 is %lu, isDescend is %lu", isInt32, isDescend);
    uint64_t schId = static_cast<uint64_t>(0);
    if (IsMergeSort(sortTileInfo)) {
        schId = static_cast<uint64_t>(0);
        GetMergeSort(context, sortTileInfo);
    } else if (IsMergeSortMultiCore(sortTileInfo)) {
        schId = SCH_ID_3;
        GetMergeSortMultiCore(context, sortTileInfo);
    } else if (IsRadixSortOneCore(sortTileInfo)) {
        schId = static_cast<uint64_t>(1);
        GetRadixSortOneCore(context, sortTileInfo);
    } else {
        schId = SCH_ID_2;
        GetRadixSortMoreCore(context, sortTileInfo);
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, isInt32, isDescend);
    OP_LOGI(context->GetNodeName(), "tilingKey is %lu, maxCoreNum %d, schId %lu", tilingKey, maxCoreNum, schId);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(sortTileInfo.coreNumNeed);
    context->SetLocalMemorySize(sortTileInfo.ubSize);
    FillTilingDataSort(sortTileInfo, sortTilingData);
    PrintTilindDataSort(context, sortTileInfo);
    OP_LOGI(context->GetNodeName(), "end RadixSortTIling ");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortTilingSimt(gert::TilingContext *context, int32_t maxCoreNum)
{
    return RadixSortTiling(context, maxCoreNum);
}

}
}