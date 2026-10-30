/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file reflection_pad1d_v2_tiling.cpp
 * \brief ReflectionPad1dV2 operator tiling implementation
 */
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/reflection_pad1d_v2_tiling_data.h"
#include "../op_kernel/reflection_pad1d_v2_tiling_key.h"

namespace optiling {

// 常量定义
const uint32_t BYTE_BLOCK = 32;    // 32字节对齐
const int32_t X_INPUT_INDEX = 0;   // 输入张量索引
const int32_t PAD_INPUT_INDEX = 1; // 填充参数索引
const int32_t OUTPUT_INDEX = 0;    // 输出张量索引

const int32_t FLOAT_BYTES = 4;   // FP32字节数
const int32_t FLOAT16_BYTES = 2; // FP16/BF16字节数

// 维度索引映射
const uint32_t DIM_1D_W = 0; // 1D: [W]
const uint32_t DIM_2D_C = 0; // 2D: [C, W]
const uint32_t DIM_2D_W = 1;
const uint32_t DIM_3D_N = 0; // 3D: [N, C, W]
const uint32_t DIM_3D_C = 1;
const uint32_t DIM_3D_W = 2;

// 填充参数索引
const uint32_t PAD_LEFT_INDEX = 0;  // 左填充
const uint32_t PAD_RIGHT_INDEX = 1; // 右填充
const uint32_t PAD_LEN = 2;         // 填充参数长度

// 硬件相关常量
const uint32_t RESERVED_UB = 32 * 1024; // 预留UB内存(32KB)
const uint32_t ALIGN_256_BYTES = 256;   // UB对齐粒度
const uint32_t MAX_CORE_NUM = 64;       // 最大核心数
const uint32_t UB_PARTITION_NUM = 5;    // UB分区数
const uint32_t MIN_UB_ELEMENTS = 32;    // 最小UB元素数

// 数据类型字节映射表
static std::map<ge::DataType, int32_t> DATATYPE_BYTES_MAP = {
    {ge::DT_FLOAT, FLOAT_BYTES}, {ge::DT_FLOAT16, FLOAT16_BYTES}, {ge::DT_BF16, FLOAT16_BYTES}};

// 向上对齐函数
template <typename T1, typename T2>
static inline T1 CeilAlign(T1 value, T2 align)
{
    if (align == 0)
        return value;
    return (value + align - 1) / align * align;
}

// 向下对齐函数
template <typename T1, typename T2>
static inline T1 FloorAlign(T1 value, T2 align)
{
    if (align == 0)
        return value;
    return value / align * align;
}

struct ReflectionPad1dV2CompileInfo {};

// Get platform information (ubSize, coreNum)
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum, uint32_t& sysWorkspaceSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTypeInfo(gert::TilingContext* context, ge::DataType& input_type, int32_t& dtypeBytes)
{
    // 解析输入张量
    const auto inputTensor = context->GetInputTensor(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputTensor);
    input_type = inputTensor->GetDataType();
    // 校验数据类型
    if (DATATYPE_BYTES_MAP.find(input_type) == DATATYPE_BYTES_MAP.end()) {
        return ge::GRAPH_FAILED;
    }
    dtypeBytes = DATATYPE_BYTES_MAP[input_type];
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, uint32_t& nSize, uint32_t& cSize, uint32_t& wSize)
{
    // 解析输入形状
    const auto inputShape = context->GetInputShape(X_INPUT_INDEX)->GetOriginShape();
    uint32_t inputDim = static_cast<uint32_t>(inputShape.GetDimNum());
    if (inputDim < 1 || inputDim > 3) {
        return ge::GRAPH_FAILED;
    }

    // 映射N/C/W维度
    nSize = 1;
    cSize = 1;
    wSize = 1;
    switch (inputDim) {
        case 1:
            wSize = static_cast<uint32_t>(inputShape[DIM_1D_W]);
            break;
        case 2:
            cSize = static_cast<uint32_t>(inputShape[DIM_2D_C]);
            wSize = static_cast<uint32_t>(inputShape[DIM_2D_W]);
            break;
        case 3:
            nSize = static_cast<uint32_t>(inputShape[DIM_3D_N]);
            cSize = static_cast<uint32_t>(inputShape[DIM_3D_C]);
            wSize = static_cast<uint32_t>(inputShape[DIM_3D_W]);
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPaddingInfo(gert::TilingContext* context, uint32_t& padLeft, uint32_t& padRight)
{
    // 解析填充参数
    const auto paddingTensor = context->GetInputTensor(PAD_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddingTensor);

    if (paddingTensor->GetDataType() == ge::DT_INT32) {
        const int32_t* padData = paddingTensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, padData);

        uint64_t padLen = paddingTensor->GetShapeSize();
        OP_CHECK_IF(padLen != PAD_LEN, OP_LOGE(context, "padLen invalid"), return ge::GRAPH_FAILED);

        int64_t left = static_cast<int64_t>(padData[PAD_LEFT_INDEX]);
        int64_t right = static_cast<int64_t>(padData[PAD_RIGHT_INDEX]);
        OP_CHECK_IF(left < 0 || right < 0, OP_LOGE(context, "pad value invalid"), return ge::GRAPH_FAILED);

        padLeft = static_cast<uint32_t>(left);
        padRight = static_cast<uint32_t>(right);
    } else if (paddingTensor->GetDataType() == ge::DT_INT64) {
        const int64_t* padData = paddingTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, padData);

        uint64_t padLen = paddingTensor->GetShapeSize();
        OP_CHECK_IF(padLen != PAD_LEN, OP_LOGE(context, "padLen invalid"), return ge::GRAPH_FAILED);

        int64_t left = static_cast<int64_t>(padData[PAD_LEFT_INDEX]);
        int64_t right = static_cast<int64_t>(padData[PAD_RIGHT_INDEX]);
        OP_CHECK_IF(left < 0 || right < 0, OP_LOGE(context, "pad value invalid"), return ge::GRAPH_FAILED);

        padLeft = static_cast<uint32_t>(left);
        padRight = static_cast<uint32_t>(right);
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(
    ReflectionPad1dV2TilingData* tilingData, uint32_t wSize, uint32_t alignWSize, uint32_t padLeft, uint32_t padRight,
    uint32_t blockNum, uint32_t ncPerCore, uint32_t tailNC, uint32_t ubElementNum, uint64_t workspacePerCore)
{
    tilingData->wSize = wSize;
    tilingData->alignWSize = alignWSize;
    tilingData->padLeft = padLeft;
    tilingData->padRight = padRight;
    tilingData->blockNum = blockNum;
    tilingData->ncPerCore = ncPerCore;
    tilingData->tailNC = tailNC;
    tilingData->ubElementNum = ubElementNum;
    tilingData->workspacePerCore = workspacePerCore;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateTilingData(
    ReflectionPad1dV2TilingData* tilingData, const uint64_t ubSizePlatForm, const uint64_t coreNum,
    const uint32_t nSize, const uint32_t cSize, const uint32_t wSize, const uint32_t padLeft, const uint32_t padRight,
    uint32_t& blockNum, uint32_t& ncPerCore, uint32_t& tailNC, const int32_t dtypeBytes, uint64_t& workspacePerCore)
{
    // 计算输出形状
    uint32_t outWSize = wSize + padLeft + padRight;
    uint32_t elePer32B = BYTE_BLOCK / dtypeBytes;
    uint32_t alignWSize = CeilAlign(wSize, elePer32B);

    // 分核策略计算
    uint64_t totalNc = static_cast<uint64_t>(nSize) * cSize;

    if (totalNc == 0 || coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    if (totalNc <= coreNum) {
        blockNum = static_cast<uint32_t>(totalNc);
        ncPerCore = 1;
        tailNC = 0;
    } else {
        ncPerCore = static_cast<uint32_t>(totalNc / coreNum);
        tailNC = static_cast<uint32_t>(totalNc % coreNum);
        blockNum = coreNum;
    }

    // UB内存分配
    uint32_t tilingDataSize = CeilAlign(sizeof(ReflectionPad1dV2TilingData), BYTE_BLOCK);
    if (ubSizePlatForm < RESERVED_UB + tilingDataSize) {
        return ge::GRAPH_FAILED;
    }

    uint32_t availableUb = 0;
    uint64_t availableUb64 = FloorAlign(ubSizePlatForm - RESERVED_UB - tilingDataSize, BYTE_BLOCK);
    if (availableUb64 > UINT32_MAX) {
        availableUb = UINT32_MAX;
    } else {
        availableUb = static_cast<uint32_t>(availableUb64);
    }

    uint32_t ubPerPartition = availableUb / UB_PARTITION_NUM;
    uint32_t ubElementNum = ubPerPartition / dtypeBytes;
    ubElementNum = FloorAlign(ubElementNum, ALIGN_256_BYTES / dtypeBytes);

    // 确保UB元素数不为0
    if (ubElementNum < MIN_UB_ELEMENTS) {
        ubElementNum = MIN_UB_ELEMENTS;
    }

    // 工作空间计算
    bool isLargePad =
        (padLeft + padRight > ubElementNum / 2) || (static_cast<uint64_t>(wSize) + padLeft + padRight > ubElementNum);

    if ((padLeft > 0 || padRight > 0) && isLargePad) {
        workspacePerCore = static_cast<uint64_t>(outWSize) * ncPerCore * dtypeBytes;
        if (tailNC > 0) {
            workspacePerCore =
                std::max(workspacePerCore, static_cast<uint64_t>(outWSize) * (ncPerCore + 1) * dtypeBytes);
        }
        workspacePerCore = CeilAlign(workspacePerCore, BYTE_BLOCK);
    }

    // Set tiling data
    SetTilingData(
        tilingData, wSize, alignWSize, padLeft, padRight, blockNum, ncPerCore, tailNC, ubElementNum, workspacePerCore);
    return ge::GRAPH_SUCCESS;
}

// Get workspace size information
static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context, uint32_t& sysWorkspaceSize, uint64_t& workspacePerCore, uint32_t& blockNum)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = workspacePerCore * blockNum + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// Tiling distribution entry
static ge::graphStatus ReflectionPad1dV2TilingFunc(gert::TilingContext* context)
{
    // Get platform runtime information
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint32_t sysWorkspaceSize = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSizePlatForm, coreNum, sysWorkspaceSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // Get type information
    ge::DataType input_type;
    int32_t dtypeBytes = 0;
    OP_CHECK_IF(
        GetTypeInfo(context, input_type, dtypeBytes) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetTypeInfo error"),
        return ge::GRAPH_FAILED);

    // Get shape information
    uint32_t nSize = 0, cSize = 0, wSize = 0;
    OP_CHECK_IF(
        GetShapeInfo(context, nSize, cSize, wSize) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeInfo error"),
        return ge::GRAPH_FAILED);

    // Get padding information
    uint32_t padLeft = 0, padRight = 0;
    OP_CHECK_IF(
        GetPaddingInfo(context, padLeft, padRight) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPaddingInfo error"),
        return ge::GRAPH_FAILED);

    // Set tiling information
    ReflectionPad1dV2TilingData* tiling = context->GetTilingData<ReflectionPad1dV2TilingData>();
    uint32_t blockNum = 0, ncPerCore = 0, tailNC = 0;
    uint64_t workspacePerCore = 0;
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ReflectionPad1dV2TilingData), 0, sizeof(ReflectionPad1dV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalculateTilingData(
            tiling, ubSizePlatForm, coreNum, nSize, cSize, wSize, padLeft, padRight, blockNum, ncPerCore, tailNC,
            dtypeBytes, workspacePerCore) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateTilingData error"), return ge::GRAPH_FAILED);

    // Get workspace size information
    OP_CHECK_IF(
        GetWorkspaceSize(context, sysWorkspaceSize, workspacePerCore, blockNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // Save tiling data and set block dimension
    context->SetBlockDim(blockNum);

    // Set tiling key based on data type
    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(input_type);
    OP_CHECK_IF(
        context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetTilingKey failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReflectionPad1dV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// Tiling registration entry
IMPL_OP_OPTILING(ReflectionPad1dV2)
    .Tiling(ReflectionPad1dV2TilingFunc)
    .InputsDataDependency({PAD_INPUT_INDEX})
    .TilingParse<ReflectionPad1dV2CompileInfo>(TilingParseForReflectionPad1dV2);
} // namespace optiling