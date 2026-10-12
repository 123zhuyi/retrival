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
 * \file non_finite_check_tiling.cpp
 * \brief
 */
#include "non_finite_check_tiling.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256;
constexpr size_t WORKSPACE_SIZE = 1;

constexpr uint8_t DTYPE_SIZE_FLOAT = 4;
constexpr uint8_t NUM_TWO = 2;
constexpr uint32_t COEFFICIENT_1 = 128;

class NonFiniteCheckTiling {
public:
    explicit NonFiniteCheckTiling(gert::TilingContext* context)
        : tilingContext(context), nodeName(context->GetNodeName()){};

    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    void InitTilingDataItems();
    ge::graphStatus CheckParams() const;
    ge::graphStatus FillCompileInfo();
    bool CalcNeedCoreNum();
    void AssignDataToEachCore();
    bool DivideUbMemory();
    uint32_t GetReduceRetValSize(uint32_t srcDataSize, uint32_t dtypeSize) const;
    uint64_t GetTilingKeyVal() const;
    void FillTilingData();

private:
    gert::TilingContext* tilingContext = nullptr;
    std::string nodeName = "NonFiniteCheck";
    NonFiniteCheckTilingData tilingData;
    NonFiniteCheckCompileInfo compileInfo;

    uint32_t maxProcCount = 0;
    uint32_t tempValUbSize = 0;
    int64_t tensorDataCountAlignedList[MAX_TENSOR_COUNT] = {0};
    int64_t* tensorDataCountList = nullptr;
    uint16_t* tensorStartList = nullptr;
    uint16_t* tensorEndList = nullptr;
    int64_t* tensorStartOffsetList = nullptr;
    int64_t* tensorEndOffsetList = nullptr;
    int64_t totalDataCountAligned = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    int32_t dataTypeSize = 0;
    int32_t elementsPerBlock = 0;
    int32_t totalTensorCount = 0;
    uint32_t needCoreNum = 0;
};

ge::graphStatus NonFiniteCheckTiling::Init()
{
    InitTilingDataItems();
    totalTensorCount = int32_t(tilingContext->GetComputeNodeInputNum());
    OP_CHECK_IF(totalTensorCount > MAX_TENSOR_COUNT || totalTensorCount <= 0,
                OP_LOGE_FOR_INVALID_TENSORNUM(tilingContext->GetNodeName(), "tensor_list", totalTensorCount,
                                              ("between 1 and " + std::to_string(MAX_TENSOR_COUNT)).c_str()),
                return ge::GRAPH_FAILED);
    // Get shape, dtype information, and the total number of data.
    for (int32_t i = 0; i < totalTensorCount; i++) {
        auto descPtr = tilingContext->GetDynamicInputDesc(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, descPtr);
        auto tempDtype = descPtr->GetDataType();
        // Determine whether all data types are consistent.
        if (i == 0) {
            dataType = tempDtype;
            dataTypeSize = ge::GetSizeByDataType(dataType);
            OP_CHECK_IF(
                dataTypeSize <= 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    tilingContext->GetNodeName(), "tesnor_list 0th tensor", Ops::Base::ToString(dataType).c_str(),
                    "The dtype size of tesnor_list 0th tensor must be greater than 0"),
                return ge::GRAPH_FAILED);
            elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        } else if (tempDtype != dataType) {
            std::string paramNames = "tensor_list " + std::to_string(i) + "th tensor";
            std::string reasonMsg = "The dtype of all input tensor_list should be the same, but tensor_list " +
                                    std::to_string(i) + "th tensor`s dtype is not same as 0th tensor`s dtype " +
                                    Ops::Base::ToString(dataType);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), paramNames.c_str(),
                                                  Ops::Base::ToString(tempDtype).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        auto shapePtr = tilingContext->GetDynamicInputShape(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, shapePtr);
        tensorDataCountList[i] = shapePtr->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(tensorDataCountList[i] == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(tilingContext->GetNodeName(),
                                                  ("tensor_list " + std::to_string(i) + "th tensor").c_str(), "0",
                                                  "greater than 0"),
                    return ge::GRAPH_FAILED);
        // Make a 32-byte alignment for each Tensor
        tensorDataCountAlignedList[i] = Ops::Base::CeilAlign(tensorDataCountList[i], int64_t(elementsPerBlock));
        totalDataCountAligned += tensorDataCountAlignedList[i];
    }

    return CheckParams();
}

ge::graphStatus NonFiniteCheckTiling::RunBigKernelTiling()
{
    OP_LOGD(tilingContext, "Start.");
    OP_CHECK_IF(FillCompileInfo() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "FillCompileInfo error."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext, "Platform info, totalCoreNum:%d, ubSizePlatForm:%lu.", compileInfo.totalCoreNum,
            compileInfo.ubSizePlatForm);
    OP_CHECK_IF(compileInfo.totalCoreNum > MAX_CORE_COUNT,
                OP_LOGE(tilingContext, "The number of totalCoreNum exceeds the limit(%hu).", MAX_CORE_COUNT),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CalcNeedCoreNum() == false, OP_LOGE(tilingContext, "Param needCoreNum is zero."),
                return ge::GRAPH_FAILED);
    AssignDataToEachCore();
    OP_CHECK_IF(DivideUbMemory() == false, OP_LOGE(tilingContext, "DivideUbMemory failed."), return ge::GRAPH_FAILED);

    FillTilingData();

    tilingContext->SetTilingKey(GetTilingKeyVal());
    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetScheduleMode(1);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;
    OP_LOGD(tilingContext, "Success.");
    return ge::GRAPH_SUCCESS;
}

void NonFiniteCheckTiling::InitTilingDataItems()
{
    tensorDataCountList = tilingData.get_tensorDataCountList();
    tensorStartList = tilingData.get_tensorStartList();
    tensorEndList = tilingData.get_tensorEndList();
    tensorStartOffsetList = tilingData.get_tensorStartOffsetList();
    tensorEndOffsetList = tilingData.get_tensorEndOffsetList();
}

ge::graphStatus NonFiniteCheckTiling::CheckParams() const
{
    OP_LOGD(tilingContext, "dataType:%d, totalTensorCount:%d, totalDataCountAligned:%ld.",
            static_cast<int32_t>(dataType), totalTensorCount, totalDataCountAligned);
    OP_CHECK_IF(dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16 && dataType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "tensor_list",
                                          ge::TypeUtils::DataTypeToSerialString(dataType).c_str(),
                                          "float16, bfloat16 or float"),
                return ge::GRAPH_FAILED);

    auto flagDescPtr = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, flagDescPtr);
    OP_CHECK_IF(
        flagDescPtr->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "found_flag",
                                  ge::TypeUtils::DataTypeToSerialString(flagDescPtr->GetDataType()).c_str(), "float"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NonFiniteCheckTiling::FillCompileInfo()
{
    auto ptrCompileInfo = tilingContext->GetCompileInfo<NonFiniteCheckCompileInfo>();
    if (ptrCompileInfo != nullptr) {
        OP_LOGD(tilingContext, "tilingContext get NonFiniteCheckCompileInfo success.");
        compileInfo = *ptrCompileInfo;
        return ge::GRAPH_SUCCESS;
    }

    auto platformInfo = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfo);

    compileInfo.totalCoreNum = int32_t(platformInfo->GetCoreNum());
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, compileInfo.ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

bool NonFiniteCheckTiling::CalcNeedCoreNum()
{
    needCoreNum = uint32_t(totalDataCountAligned / elementsPerBlock);
    if (needCoreNum > uint32_t(compileInfo.totalCoreNum)) {
        needCoreNum = compileInfo.totalCoreNum;
    }
    if (needCoreNum == 0) {
        return false;
    } else {
        return true;
    }
}

void NonFiniteCheckTiling::AssignDataToEachCore()
{
    int64_t blockCount = totalDataCountAligned / elementsPerBlock;
    // Divisible, representing the amount of data each core needs to process.
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum; // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (uint16_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount != 0 && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountAlignedList[i] - cursorPos;
        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountAlignedList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPos;
            --i; // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount != 0) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountAlignedList[totalTensorCount - 1] - 1;
    }
}

bool NonFiniteCheckTiling::DivideUbMemory()
{
    // A 32-byte alignment is performed on the UB available space.
    uint32_t canUseUbSize = uint32_t(compileInfo.ubSizePlatForm / BYTE_BLOCK * BYTE_BLOCK);
    uint32_t dtypeSizeTemp = dataTypeSize;
    if (dataType == ge::DT_BF16) {
        dtypeSizeTemp = DTYPE_SIZE_FLOAT;
    }
    uint32_t predictSGUbSize = uint32_t((canUseUbSize - BYTE_BLOCK) * COEFFICIENT_1 * 1.0 /
                                        (BYTE_REPEAT + dtypeSizeTemp));
    uint32_t maxDataUbSize = predictSGUbSize / BYTE_BLOCK * BYTE_BLOCK;
    maxProcCount = maxDataUbSize / dtypeSizeTemp;
    tempValUbSize = GetReduceRetValSize(maxDataUbSize, dtypeSizeTemp);
    if ((NUM_TWO * maxDataUbSize + tempValUbSize) > compileInfo.ubSizePlatForm) {
        return false;
    } else {
        return true;
    }
}

uint32_t NonFiniteCheckTiling::GetReduceRetValSize(uint32_t srcDataSize, uint32_t dtypeSize) const
{
    /* Calculate the space size of the intermediate variable workLocal and
        the result variable dstLocal of ReduceMax and ReduceMin. */
    uint8_t perBlockCount = BYTE_BLOCK / dtypeSize;
    uint32_t iter1OutputCount = uint32_t(std::ceil(NUM_TWO * 1.0 * srcDataSize / BYTE_REPEAT));
    uint32_t iter1AlignEnd = Ops::Base::CeilAlign(iter1OutputCount, uint32_t(perBlockCount));
    return iter1AlignEnd * dtypeSize;
}

uint64_t NonFiniteCheckTiling::GetTilingKeyVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_FLOAT);
        case ge::DT_FLOAT16:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_FLOAT16);
        case ge::DT_BF16:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_BF16);
        default:
            return 0;
    }
}

void NonFiniteCheckTiling::FillTilingData()
{
    std::ostringstream tempOSS;
    tempOSS << "maxProcCount: " << maxProcCount << ", tempValUbSize: " << tempValUbSize << "." << std::endl
            << "tensorDataCountList: ";
    for (uint32_t i = 0; i < uint32_t(totalTensorCount); i++) {
        tempOSS << "[" << i << "]: " << tensorDataCountList[i] << ", ";
    }
    tempOSS << std::endl << "Per core info: " << std::endl;
    for (uint32_t i = 0; i < needCoreNum; i++) {
        tempOSS << "index:[" << i << "], tensorStartList: " << tensorStartList[i]
                << ", tensorEndList:" << tensorEndList[i] << ", tensorStartOffsetList: " << tensorStartOffsetList[i]
                << ", tensorEndOffsetList: " << tensorEndOffsetList[i] << "." << std::endl;
    }

    OP_LOGD(tilingContext, "Tiling data, %s", tempOSS.str().c_str());

    tilingData.set_maxProcCount(maxProcCount);
    tilingData.set_tempValUbSize(tempValUbSize);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus Tiling4NonFiniteCheck(gert::TilingContext* context)
{
    NonFiniteCheckTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Init tiling object return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.RunBigKernelTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Run big kernel tiling return failed.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4NonFiniteCheck(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<NonFiniteCheckCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0),
                OP_LOGE(context, "TilingPrepare4NonFiniteCheck get aiv core num failed."), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0),
                OP_LOGE(context, "TilingPrepare4NonFiniteCheck get ub size failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NonFiniteCheck)
    .Tiling(Tiling4NonFiniteCheck)
    .TilingParse<NonFiniteCheckCompileInfo>(TilingPrepare4NonFiniteCheck);

} // namespace optiling