/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
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
 * \file eye_v2_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/eye_v2_tiling_data.h"
#include "../op_kernel/eye_v2_tiling_key.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 512U;

struct EyeV2CompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum, uint32_t& blockSize)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, ge::DataType& dataType)
{
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_FLOAT16};
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    dataType = outputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreNum(
    gert::TilingContext* context, EyeV2TilingData* tiling, int64_t coreNum, uint32_t blockSize)
{
    // 获取输入张量形状和数据类型
    uint32_t typeLength = 0;
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    ge::TypeUtils::GetDataTypeLength(outputDesc->GetDataType(), typeLength);
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    // 读取属性rows, cols
    int32_t rows = 0;
    int32_t cols = 0;
    auto attrs = context->GetAttrs();
    if (attrs) {
        const int64_t* rowsPtr = attrs->GetInt(0);
        if (rowsPtr) {
            rows = static_cast<int32_t>(*rowsPtr);
        }
        const int64_t* colsPtr = attrs->GetInt(1);
        if (colsPtr) {
            cols = static_cast<int32_t>(*colsPtr);
        }
    }
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    uint64_t typeSize = inputBytes;
    uint64_t alignNum = blockSize / typeSize;
    // 核间-计算每个核处理的对角线长度
    uint64_t diagLen = std::min<uint64_t>(static_cast<uint64_t>(rows), static_cast<uint64_t>(cols));
    uint64_t usableCoreNum =
        (diagLen == 0) ? 1U : std::min<uint64_t>(static_cast<uint64_t>(coreNum), static_cast<uint64_t>(diagLen));
    uint64_t tailBlockLength = diagLen / usableCoreNum;
    uint64_t fullBlockNum = diagLen % usableCoreNum;
    uint64_t tailBlockNum = usableCoreNum - fullBlockNum;
    uint64_t fullBlockLength = tailBlockLength + 1;

    tiling->typeSize = typeSize;
    tiling->alignNum = alignNum;
    tiling->matrixOrder = 0;
    tiling->rowLength = rows;
    tiling->columnLength = cols;
    tiling->diagLen = diagLen;
    tiling->fullBlockLength = fullBlockLength;
    tiling->tailBlockLength = tailBlockLength;
    tiling->fullBlockNum = fullBlockNum;
    tiling->tailBlockNum = tailBlockNum;
    context->SetBlockDim(usableCoreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus EyeV2TilingFunc(gert::TilingContext* context)
{
    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    uint32_t blockSize = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum, blockSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. shapes & dtype
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    EyeV2TilingData* tiling = context->GetTilingData<EyeV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(EyeV2TilingData), 0, sizeof(EyeV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 4. CoreNum & Tiling Calculation
    OP_CHECK_IF(
        CalculateCoreNum(context, tiling, coreNum, blockSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateCoreNum error"), return ge::GRAPH_FAILED);
    context->GetRawTilingData()->SetDataSize(sizeof(EyeV2TilingData));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForEyeV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
// tiling注册入口.
IMPL_OP_OPTILING(EyeV2).Tiling(EyeV2TilingFunc).TilingParse<EyeV2CompileInfo>(TilingParseForEyeV2);
} // namespace optiling
