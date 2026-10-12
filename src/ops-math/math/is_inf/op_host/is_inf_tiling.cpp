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
 * \file is_inf_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "is_inf_tiling_arch35.h"
#include "is_inf_tiling_def.h"
#include "log/log.h"
#include "op_host/tiling_base_util.h"

namespace optiling {

constexpr uint32_t DATA_BLOCK = 32;
constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_BFLOAT16 = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;
constexpr uint32_t MAX_DIM = 8;

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

class IsInfTiling
{
public:
    explicit IsInfTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunBigKernelTiling();

private:
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    void AssignDataToEachCore();
    void FillTilingData();

    uint8_t GetDataTypeSize() const;
    uint64_t GetTilingKeyVal() const;

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform) const;
    uint32_t GetUsableUbMemory(uint64_t ubSizePlatForm);

private:
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;
    IsInfTilingData tilingData;

    uint8_t dataBlockSize = 0;

    uint32_t totalDataCount = 1;
    uint32_t usableUbSize = 0;
    uint32_t needCoreNum = 0;
    uint32_t perCoreDataCount = 0;
    uint32_t tailDataCoreNum = 0;
    uint32_t lastCoreDataCount = 0;
};

ge::graphStatus IsInfTiling::RunBigKernelTiling()
{
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

    uint64_t ubSizePlatForm = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    // Get dtype information, and the total number of data.
    if (tilingContext != nullptr && tilingContext->GetInputDesc(0) != nullptr) {
        dataType = tilingContext->GetInputDesc(0)->GetDataType();
    }
    uint8_t dataTypeSize = GetDataTypeSize();
    dataBlockSize = DATA_BLOCK * dataTypeSize;
    const gert::StorageShape* shape = tilingContext->GetInputShape(0);
    uint16_t dimNumOfShape = 0;
    if (shape != nullptr) {
        dimNumOfShape = shape->GetStorageShape().GetDimNum();
        if (dimNumOfShape > MAX_DIM){
            OP_LOGW("IsInfTilingDimCheck","Input shape has %u dimensions, which exceeds the recommended maximum of %u dimensions.",dimNumOfShape,MAX_DIM);
        }
    }
    for (uint16_t i = 0; i < dimNumOfShape; i++) {
        if (shape != nullptr) {
            totalDataCount *= shape->GetStorageShape().GetDim(i);
        }
    }

    usableUbSize = GetUsableUbMemory(ubSizePlatForm);
    needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    AssignDataToEachCore();
    FillTilingData();

    tilingContext->SetTilingKey(GetTilingKeyVal());
    tilingContext->SetBlockDim(needCoreNum);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = WORK_SPACE_SIZE;
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

template <typename T1, typename T2>
inline T1 IsInfTiling::CeilA2B(T1 a, T2 b) const
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

uint8_t IsInfTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

uint64_t IsInfTiling::GetTilingKeyVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_BF16:
            return TILING_KEY_BFLOAT16;
        default:
            return 0;
    }
}

uint32_t IsInfTiling::GetNeedCoreNum(uint32_t coreNumPlatform) const
{
    uint32_t tempCoreNum = (uint32_t)CeilA2B(totalDataCount, DATA_BLOCK);
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void IsInfTiling::AssignDataToEachCore()
{
    perCoreDataCount = totalDataCount / needCoreNum;
    perCoreDataCount = perCoreDataCount / DATA_BLOCK * DATA_BLOCK;
    uint32_t tempTailDataCount = totalDataCount - perCoreDataCount * needCoreNum;
    tailDataCoreNum = tempTailDataCount / DATA_BLOCK;
    lastCoreDataCount = perCoreDataCount + tempTailDataCount % DATA_BLOCK;
}

uint32_t IsInfTiling::GetUsableUbMemory(uint64_t ubSizePlatForm)
{
    // The remaining UB size is split in two, double buffer enabled, input and output, and rounded down 32 data.
    uint32_t canUseUbSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize()) / UB_DIVIDER_FOR_TEMP_CASTING;
    canUseUbSize = canUseUbSize / dataBlockSize * dataBlockSize;
    return canUseUbSize;
}

void IsInfTiling::FillTilingData()
{
    tilingData.set_totalDataCount(totalDataCount);
    tilingData.set_usableUbSize(usableUbSize);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_perCoreDataCount(perCoreDataCount);
    tilingData.set_tailDataCoreNum(tailDataCoreNum);
    tilingData.set_lastCoreDataCount(lastCoreDataCount);
}

static ge::graphStatus TilingPrepare4IsInfTiling([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingIsInfTiling(gert::TilingContext* context)
{
    if (Ops::Base::IsRegbaseSocVersion(context)) {
        OP_LOGD("IsInfTiling", "Enter new IsInfTiling");

        fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

        uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
        OP_CHECK_IF(
            (static_cast<int32_t>(coreNum) <= 0), OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core_num", std::to_string(coreNum), "Failed to get core num"), return ge::GRAPH_FAILED);

        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(
            (static_cast<int64_t>(ubSize) <= 0), OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ub_size", std::to_string(ubSize), "Failed to get ub size"), return ge::GRAPH_FAILED);

        IsInfRegbaseTiling IsInfRegbaseTiling(context);
        return IsInfRegbaseTiling.RunTiling();
    }
    IsInfTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(IsInf).Tiling(TilingIsInfTiling).TilingParse<IsInfCompileInfo>(TilingPrepare4IsInfTiling);
} // namespace optiling
