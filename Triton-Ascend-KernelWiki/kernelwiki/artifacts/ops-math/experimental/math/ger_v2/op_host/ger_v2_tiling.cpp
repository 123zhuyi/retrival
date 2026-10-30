/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
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
 * \file ger_v2_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/ger_v2_tiling_data.h"
#include "../op_kernel/ger_v2_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

const int32_t DIM_X = 1; // 期望 x 的维度
const int32_t DIM_Y = 1; // 期望 y 的维度
const int32_t DIM_A = 2; // 期望 A 的维度
const int32_t DIM_Z = 2; // 期望 Z 的维度

static constexpr int32_t IDX_0 = 0;
static constexpr int32_t IDX_1 = 1;
static constexpr int32_t OUT_DIMS = 2;
static constexpr int32_t OUT_DIM_0 = 0U;
static constexpr int32_t OUT_DIM_1 = 1U;

constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr uint32_t INDEXTWO = 2;
constexpr uint32_t INDEXTHREE = 3;

struct GerV2CompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint32_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& m, int64_t& n, ge::DataType& dataType)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
    auto inputShapeX = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());

    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = Ops::Base::EnsureNotScalar(inputY->GetStorageShape());

    auto inputA = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputA);
    auto inputShapeA = Ops::Base::EnsureNotScalar(inputA->GetStorageShape());

    auto outputZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputZ);
    auto outputShapeZ = Ops::Base::EnsureNotScalar(outputZ->GetStorageShape());

    // shape校验
    // ===== Step 1: rank 校验（必须先做，避免 GetDim 越界）=====
    OP_CHECK_IF(
        inputShapeX.GetDimNum() != DIM_X || inputShapeY.GetDimNum() != DIM_Y || inputShapeA.GetDimNum() != DIM_A ||
            outputShapeZ.GetDimNum() != DIM_Z,
        OP_LOGE(
            context,
            "Ger: rank mismatch. "
            "x(rank=%zu), y(rank=%zu), A(rank=%zu), Z(rank=%zu)",
            inputShapeX.GetDimNum(), inputShapeY.GetDimNum(), inputShapeA.GetDimNum(), outputShapeZ.GetDimNum()),
        return ge::GRAPH_FAILED);

    // ===== Step 2: rank 已验证安全，可以 GetDim =====
    m = inputShapeX.GetDim(IDX_0); // x 长度
    n = inputShapeY.GetDim(IDX_0); // y 长度

    auto A_dim0 = inputShapeA.GetDim(IDX_0);
    auto A_dim1 = inputShapeA.GetDim(IDX_1);

    auto Z_dim0 = outputShapeZ.GetDim(IDX_0);
    auto Z_dim1 = outputShapeZ.GetDim(IDX_1);

    // ===== Step 3: shape 大小校验 =====
    OP_CHECK_IF(
        A_dim0 != m || A_dim1 != n || Z_dim0 != m || Z_dim1 != n,
        OP_LOGE(
            context,
            "Ger: shape check failed. "
            "x(rank=%zu, m=%ld), y(rank=%zu, n=%ld), "
            "A(rank=%zu, [%ld,%ld]), Z(rank=%zu, [%ld,%ld]) | "
            "expected x(1-D), y(1-D), A/Z(2-D [m,n]).",
            inputShapeX.GetDimNum(), m, inputShapeY.GetDimNum(), n, inputShapeA.GetDimNum(), A_dim0, A_dim1,
            outputShapeZ.GetDimNum(), Z_dim0, Z_dim1),
        return ge::GRAPH_FAILED);

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
// 可直接替换你的 GerV2TilingFunc 内部实现（保留函数签名）
static ge::graphStatus GerV2TilingFunc(gert::TilingContext* context)
{
    // 1. platform
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. shapes & dtype

    int64_t m = 0;
    int64_t n = 0;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, m, n, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    GerV2TilingData* tiling = context->GetTilingData<GerV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(GerV2TilingData), 0, sizeof(GerV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    float alpha = 1.0f; // fallback
    if (attrs) {
        const float* attrA = attrs->GetFloat(0); // 只有一个属性，索引 0
        if (attrA != nullptr) {
            alpha = *attrA;
        }
    }

    // --- safer numeric types ---
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputBytes = static_cast<uint64_t>(typeLength); // 数据类型长度
    uint32_t ubDataNumber = 5U;

    uint32_t tileDataNum = ubSize / BUFFER_NUM / ubDataNumber / inputBytes;
    if (tileDataNum == 0U)
        tileDataNum = 1U;
    int64_t tileNums = n / static_cast<int64_t>(tileDataNum); // 一行分为几个tile
    int64_t tailTileDataNum = n % static_cast<int64_t>(tileDataNum);
    tileNums = (tailTileDataNum == 0) ? tileNums : tileNums + 1;
    tailTileDataNum = (tailTileDataNum == 0) ? tileDataNum : tailTileDataNum;

    coreNum = (static_cast<uint32_t>(m) < coreNum) ? static_cast<uint32_t>(m) : coreNum; // 调用aicore数量
    int64_t smallCoreRows = m / static_cast<int64_t>(coreNum);
    int64_t tailRowNum = m % static_cast<int64_t>(coreNum); // 大核数量
    int64_t bigCoreRows = smallCoreRows + 1;

    tiling->smallCoreRows = smallCoreRows;
    tiling->bigCoreRows = bigCoreRows;
    tiling->tailRowNum = tailRowNum;
    tiling->tileNums = tileNums;
    tiling->tileDataNum = tileDataNum;
    tiling->tailTileDataNum = tailTileDataNum;
    tiling->m = m;
    tiling->n = n;
    tiling->alpha = alpha;

    context->SetBlockDim(coreNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGerV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(GerV2).Tiling(GerV2TilingFunc).TilingParse<GerV2CompileInfo>(TilingParseForGerV2);
} // namespace optiling