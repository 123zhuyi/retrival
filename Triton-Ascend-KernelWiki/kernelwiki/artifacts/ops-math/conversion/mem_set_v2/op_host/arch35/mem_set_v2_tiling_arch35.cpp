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
 * \file mem_set_v2_tiling_arch35.cpp
 * \brief tiling for mem set v2
 */

#include "mem_set_v2_tiling_arch35.h"

namespace optiling {
using namespace ge;
std::set<ge::DataType> g_intOrBoolDtypesSet = {DT_INT8,   DT_INT32, DT_UINT8,  DT_INT16, DT_UINT16,
                                               DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL};
std::set<ge::DataType> g_floatDtypesSet = {DT_FLOAT, DT_FLOAT16, DT_BF16};
constexpr uint64_t FILL_WORKSPACE_RESERVE_BYTE = 16777216; // 16 * 1024 * 1024
constexpr uint32_t DTYPE_INT = 0;
constexpr uint32_t DTYPE_FLOAT = 1;
constexpr uint64_t BLOCK_SIZE = 65536;
constexpr uint32_t CUSTOM_KEY_VALUE = 99;

template <typename T>
void PrintTilingData(T& tilingData, const uint64_t& tilingKey)
{
    OP_LOGI(
        "Mem set v2 tilingData",
        "[tilingKey]: %ld, [realCoreNum]: %ld, [ubSize]: %ld, [processGMNum]: %ld, [valuesIntListSize]: %ld, "
        "[valuesFloatListSize]: %ld",
        tilingKey, tilingData.get_realCoreNum(), tilingData.get_ubSize(), tilingData.get_processGMNum(),
        tilingData.get_valuesIntListSize(), tilingData.get_processGMNum() - tilingData.get_valuesIntListSize());
    std::string intString;
    for (uint32_t i = 0; i < tilingData.get_processGMNum(); i++) {
        intString += std::to_string(tilingData.get_valuesIntList()[i]) + ", ";
    }
    std::string floatString;
    for (uint32_t i = 0; i < tilingData.get_processGMNum(); i++) {
        floatString += std::to_string(tilingData.get_valuesFloatList()[i]) + ", ";
    }
    std::string blockNumListString;
    for (uint32_t i = 0; i < tilingData.get_processGMNum(); i++) {
        blockNumListString += std::to_string(tilingData.get_blockNumList()[i]) + ", ";
    }
    std::string tailSizeListString;
    for (uint32_t i = 0; i < tilingData.get_processGMNum(); i++) {
        tailSizeListString += std::to_string(tilingData.get_tailSizeList()[i]) + ", ";
    }
    std::string startIndexListString;
    for (uint32_t i = 0; i < tilingData.get_processGMNum(); i++) {
        startIndexListString += std::to_string(tilingData.get_startIndexList()[i]) + ", ";
    }
    OP_LOGI("Mem set v2 tilingData", "[valuesIntList]: [%s]", intString.c_str());
    OP_LOGI("Mem set v2 tilingData", "[valuesFloatList]: [%s]", floatString.c_str());
    OP_LOGI("Mem set v2 tilingData", "[blockNumList]: [%s]", blockNumListString.c_str());
    OP_LOGI("Mem set v2 tilingData", "[tailSizeList]: [%s]", tailSizeListString.c_str());
    OP_LOGI("Mem set v2 tilingData", "[startIndexList]: [%s]", startIndexListString.c_str());
}

template <typename T>
ge::graphStatus SetDtypesList(
    T& tilingData, gert::TilingContext* context, int64_t& valuesIntSizeInDtypesList,
    int64_t& valuesFloatSizeInDtypesList, const int64_t inputXNum)
{
    // 计算dtypesList中int和float的个数,并转换DataType为int64_t
    vector<int64_t> dtypesArray(inputXNum, 0);
    for (int64_t i = 0; i < inputXNum; i++) {
        ge::DataType dtypeInt = context->GetDynamicInputTensor(0, i)->GetDataType();
        if (g_intOrBoolDtypesSet.find(dtypeInt) != g_intOrBoolDtypesSet.end()) {
            valuesIntSizeInDtypesList++;
        } else if (g_floatDtypesSet.find(dtypeInt) != g_floatDtypesSet.end()) {
            valuesFloatSizeInDtypesList++;
        } else {
            OP_CHECK_IF(
                true, OP_LOGE(context->GetNodeName(), "dtype should be int, float or bool."), return ge::GRAPH_FAILED);
        }
        dtypesArray[i] = dtypeInt;
    }
    tilingData.set_dtypesList(dtypesArray.data());
    return ge::GRAPH_SUCCESS;
}

uint32_t GetIntOrFloatDTypeByGmIndex(const uint32_t& dtype)
{
    if (dtype == ge::DataType::DT_FLOAT16 || dtype == ge::DataType::DT_FLOAT || dtype == ge::DataType::DT_BF16) {
        return DTYPE_FLOAT;
    } else {
        return DTYPE_INT;
    }
}

template <typename T>
ge::graphStatus SetValuesLists(
    T& tilingData, gert::TilingContext* context, const int64_t valuesIntSizeInDtypesList,
    const int64_t valuesFloatSizeInDtypesList)
{
    auto* attrs = context->GetAttrs();

    // 校验valuesInt_array和valuesFloat_array的size是否为0或者等于dtypesList中int和float的个数
    const auto* valuesIntArrayInAttrs = attrs->GetListInt(INDEX_VALUESINT);
    const auto* valuesFloatArrayInAttrs = attrs->GetListFloat(INDEX_VALUESFLOAT);
    OP_CHECK_IF(
        valuesIntArrayInAttrs == nullptr || valuesFloatArrayInAttrs == nullptr,
        OP_LOGE(context->GetNodeName(), "valuesIntArrayInAttrs or valuesFloatArrayInAttrs is nullptr."),
        return ge::GRAPH_FAILED);
    int64_t valuesIntArraySize = valuesIntArrayInAttrs->GetCapacity();
    int64_t valuesFloatArraySize = valuesFloatArrayInAttrs->GetCapacity();
    OP_CHECK_IF(
        valuesIntArraySize != 0 && valuesIntArraySize != valuesIntSizeInDtypesList,
        OP_LOGE(
            context->GetNodeName(), "ValuesInt size should be zero or the same as integer type num in tensor list."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        valuesFloatArraySize != 0 && valuesFloatArraySize != valuesFloatSizeInDtypesList,
        OP_LOGE(
            context->GetNodeName(), "ValuesFloat size should be zero or the same as float type num in tensor list."),
        return ge::GRAPH_FAILED);
    if (valuesIntArraySize == 0 && valuesIntSizeInDtypesList > 0) {
        OP_LOGW("Mem set v2 tilingData", "valuesIntArraySize is 0, fill with value 0.");
    }
    if (valuesFloatArraySize == 0 && valuesFloatSizeInDtypesList > 0) {
        OP_LOGW("Mem set v2 tilingData", "valuesFloatArraySize is 0, fill with value 0.0.");
    }
    // 如果valuesInt_array_size为0,则valuesIntList_填充为0；如果valuesFloat_array_size为0,则valuesFloatList_填充为0.0
    vector<int64_t> valuesIntArray(valuesIntSizeInDtypesList + valuesFloatSizeInDtypesList, 0);
    vector<float> valuesFloatArray(valuesIntSizeInDtypesList + valuesFloatSizeInDtypesList, 0.0);
    uint32_t typeIntNum = 0;
    uint32_t typeFloatNum = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(valuesIntSizeInDtypesList + valuesFloatSizeInDtypesList); i++) {
        if (GetIntOrFloatDTypeByGmIndex(tilingData.get_dtypesList()[i]) == DTYPE_INT && valuesIntArraySize > 0) {
            valuesIntArray[i] = valuesIntArrayInAttrs->GetData()[typeIntNum];
            typeIntNum++;
        }
        if (GetIntOrFloatDTypeByGmIndex(tilingData.get_dtypesList()[i]) != DTYPE_INT && valuesFloatArraySize > 0) {
            valuesFloatArray[i] = valuesFloatArrayInAttrs->GetData()[typeFloatNum];
            typeFloatNum++;
        }
    }
    // 设置tilingData
    tilingData.set_valuesIntList(valuesIntArray.data());
    tilingData.set_valuesFloatList(valuesFloatArray.data());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus SetBlockInfoList(
    T& tilingData, gert::TilingContext* context, const int64_t inputXNum, const uint64_t coreNum)
{
    vector<uint64_t> blockNumList(inputXNum, 0);
    vector<uint64_t> tailSizeList(inputXNum, 0);
    vector<uint64_t> startIndexList(inputXNum, 0);
    for (uint32_t i = 0; i < static_cast<uint32_t>(inputXNum); i++) {
        uint32_t dataSize = ge::GetSizeByDataType(context->GetDynamicInputTensor(0, i)->GetDataType());
        auto valueShape = Ops::Base::EnsureNotScalar(context->GetDynamicInputTensor(0, i)->GetStorageShape());
        uint64_t dataNum = valueShape.GetShapeSize();
        uint64_t sizeByte = dataNum * dataSize;
        tailSizeList[i] = sizeByte % BLOCK_SIZE;
        uint32_t tailOrNot = tailSizeList[i] > 0 ? 1 : 0;
        blockNumList[i] = sizeByte / BLOCK_SIZE + tailOrNot;
        if (i == 0U) {
            startIndexList[i] = 0;
        } else {
            OP_CHECK_IF(
                coreNum == 0,
                OP_LOGE(
                    context->GetNodeName(),
                    "ValuesFloat size should be zero or the same as float type num in tensor list."),
                return ge::GRAPH_FAILED);
            startIndexList[i] = (startIndexList[i - 1] + blockNumList[i - 1]) % coreNum;
        }
    }
    tilingData.set_blockNumList(blockNumList.data());
    tilingData.set_tailSizeList(tailSizeList.data());
    tilingData.set_startIndexList(startIndexList.data());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus DoTiling(gert::TilingContext* context, uint64_t tilingKey, int64_t inputXNum)
{
    T tilingData;
    int64_t valuesIntSizeInDtypesList = 0;
    int64_t valuesFloatSizeInDtypesList = 0;
    OP_CHECK_IF(
        SetDtypesList(tilingData, context, valuesIntSizeInDtypesList, valuesFloatSizeInDtypesList, inputXNum) ==
            ge::GRAPH_FAILED,
        OP_LOGE(context->GetNodeName(), "SetDtypesList failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SetValuesLists(tilingData, context, valuesIntSizeInDtypesList, valuesFloatSizeInDtypesList) == ge::GRAPH_FAILED,
        OP_LOGE(context->GetNodeName(), "SetValuesLists failed."), return ge::GRAPH_FAILED);

    tilingData.set_processGMNum(inputXNum);
    tilingData.set_valuesIntListSize(valuesIntSizeInDtypesList);

    uint64_t ubSize = 0;
    uint64_t coreNum = 0;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        OP_LOGD(context->GetNodeName(), "Get compileInfo without tilingParse.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        OP_LOGD(context->GetNodeName(), "Get compileInfo with tilingParse.");
        auto compileInfo = context->GetCompileInfo<MemSetV2CompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        ubSize = compileInfo->ubSize;
        coreNum = compileInfo->coreNum;
    }
    OP_CHECK_IF(
        (coreNum <= SIZE_ZERO),
        OP_LOGE(context->GetNodeName(), "MemSetV2Op GetHardwareInfo Failed, coreNum:%lu", coreNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ubSize <= SIZE_ZERO, OP_LOGE(context->GetNodeName(), "MemSetV2Op GetHardwareInfo Failed, ubSize:%lu.", ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu.", coreNum, ubSize);

    OP_CHECK_IF(
        SetBlockInfoList(tilingData, context, inputXNum, coreNum) == ge::GRAPH_FAILED,
        OP_LOGE(context->GetNodeName(), "SetBlockInfoList failed."), return ge::GRAPH_FAILED);

    tilingData.set_realCoreNum(coreNum);
    tilingData.set_ubSize(ubSize);
    // 打印tilingData
    PrintTilingData(tilingData, tilingKey);
    // 保存tilingData
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = FILL_WORKSPACE_RESERVE_BYTE;

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(tilingData.get_realCoreNum());
    OP_LOGI("Mem set v2 tilingData", "End tiling for MemSetV2.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4MemSetV2(gert::TilingContext* context)
{
    OP_LOGI("MemSetV2 tilingData", "Start tiling for MemSetV2.");
    // 根据input判断tilingkey和tilingData
    int64_t inputXNum = context->GetIrInputInstanceInfo(0)->GetInstanceNum();
    OP_CHECK_IF(
        inputXNum <= 0 || inputXNum > 256, OP_LOGE(context->GetNodeName(), "dynamic input num should in [1, 256]."),
        return ge::GRAPH_FAILED);
    if (inputXNum <= TENSOR_LIST_SIZE_2) {
        return DoTiling<MemSetV2List2TilingData>(context, TILINGKEY_10002, inputXNum);
    }
    if (inputXNum <= TENSOR_LIST_SIZE_4) {
        return DoTiling<MemSetV2List4TilingData>(context, TILINGKEY_10004, inputXNum);
    }
    if (inputXNum <= TENSOR_LIST_SIZE_8) {
        return DoTiling<MemSetV2List8TilingData>(context, TILINGKEY_10008, inputXNum);
    }
    if (inputXNum <= TENSOR_LIST_SIZE_16) {
        return DoTiling<MemSetV2List16TilingData>(context, TILINGKEY_10016, inputXNum);
    }
    if (inputXNum <= TENSOR_LIST_SIZE_32) {
        return DoTiling<MemSetV2List32TilingData>(context, TILINGKEY_10032, inputXNum);
    }
    if (inputXNum <= TENSOR_LIST_SIZE_64) {
        return DoTiling<MemSetV2List64TilingData>(context, TILINGKEY_10064, inputXNum);
    }
    if (inputXNum <= TENSOR_LIST_SIZE_128) {
        return DoTiling<MemSetV2List128TilingData>(context, TILINGKEY_10128, inputXNum);
    }
    return DoTiling<MemSetV2List256TilingData>(context, TILINGKEY_10256, inputXNum);
}

ge::graphStatus TilingPrepare4MemSetV2(gert::TilingParseContext* context)
{
    OP_LOGI("MemSetV2 tilingData", "Start tiling prepare for MemSetV2.");
    auto compileInfo = context->GetCompiledInfo<MemSetV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= SIZE_ZERO,
        OP_LOGE(context->GetNodeName(), "MemSetV2Op GetHardwareInfo Failed, ubSize:%lu.", compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "MemSetV2Op GetHardwareInfo Failed, coreNum:%lu", compileInfo->coreNum),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu.", compileInfo->coreNum, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GenSimplifiedKey4MemSetV2(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    OP_LOGD("MemSetV2", "Enter MemSetV2 genSimplifiedKey.");

    constexpr size_t DEST_MAX = 100;
    constexpr size_t MAX_LEN_SIMPLIFIED_KEY = 256;

    OP_CHECK_IF(
        simplifiedKey == nullptr, OP_LOGE("MemSetV2", "GenSimplifiedKey4MemSetV2 SimplifiedKey is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("MemSetV2", "GenSimplifiedKey4MemSetV2 Context is null"), return ge::GRAPH_FAILED);

    std::string simpleKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,99");
    OP_LOGW(context->GetNodeName(), "SimpleKeyTemp: %s", simpleKeyTemp.c_str());
    errno_t err = strcat_s(simplifiedKey, DEST_MAX, simpleKeyTemp.c_str());
    OP_CHECK_IF(
        (err != 0), OP_LOGE(context->GetNodeName(), "Error: strcat_s failed with error code %d.", err),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY,
        OP_LOGE(context->GetNodeName(), "len of simplifiedKey exceeds max length."), return ge::GRAPH_FAILED);
    OP_LOGD("MemSetV2", "Finish MemSetV2 genSimplifiedKey.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MemSetV2)
    .Tiling(Tiling4MemSetV2)
    .TilingParse<MemSetV2CompileInfo>(TilingPrepare4MemSetV2)
    .GenSimplifiedKey(GenSimplifiedKey4MemSetV2);
} // namespace optiling