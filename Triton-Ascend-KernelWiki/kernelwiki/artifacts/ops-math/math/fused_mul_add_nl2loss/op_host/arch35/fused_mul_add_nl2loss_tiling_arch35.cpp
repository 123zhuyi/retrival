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
 * \file fused_mul_add_nl2loss_tiling_arch35.cpp
 * \brief FusedMulAddNL2loss arch35 tiling
 *        展平为一维，按核均分；不使用 workspace
 *        （y2 归约：fp32 走 y2 GM 原子加，fp16 走 core0 串行，见 kernel 注释）
 */

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "fused_mul_add_nl2loss_tiling_arch35.h"
#include "log/log.h"

using namespace optiling;

// asc_opc 编译需要注册 tiling 结构体（字段与 op_kernel 侧 plain struct 二进制一致）
BEGIN_TILING_DATA_DEF(FusedMulAddNL2lossTilingDataDef)
TILING_DATA_FIELD_DEF(uint64_t, totalElements);
TILING_DATA_FIELD_DEF(uint64_t, coreElements);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreElements);
TILING_DATA_FIELD_DEF(uint64_t, ubTileSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedMulAddNL2loss, FusedMulAddNL2lossTilingDataDef)

using namespace ge;

namespace optiling {

static constexpr int64_t VL = 64;       // fp32 向量寄存器宽度
static constexpr int64_t QUEUE_NUM = 3; // x1_in, x2_in, y1_out（双缓冲）
static constexpr int64_t DOUBLE_BUFFER = 2;

static constexpr int64_t FLOAT_BYTES = 4;
static constexpr int64_t RESERVED_UB = 8512; // TPipe 元数据 + ReduceSum 小 buffer(320B) + 余量

ge::graphStatus FusedMulAddNL2lossTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfo = reinterpret_cast<const FusedMulAddNL2lossCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    }
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_, "invalid coreNum %ld", coreNum_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= 0, OP_LOGE(context_, "invalid ubSize %ld", ubSize_), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMulAddNL2lossTiling::DoTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 取 x1 shape，展平为 N
    auto inputX1Shape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1Shape);
    auto shapeX1 = inputX1Shape->GetStorageShape();
    int64_t totalN = shapeX1.GetShapeSize();
    OP_CHECK_IF(totalN <= 0, OP_LOGE(context_, "invalid total elements %ld", totalN), return ge::GRAPH_FAILED);

    // 对齐 910b verifier：x2 必须与 x1 元素数一致（kernel 按同一 N 索引 x2）
    auto inputX2Shape = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX2Shape);
    int64_t totalX2 = inputX2Shape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalX2 != totalN, OP_LOGE(context_, "x2 elements %ld != x1 elements %ld", totalX2, totalN),
                return ge::GRAPH_FAILED);

    // UB tile 宽度：fp32 路径 3 队列×双缓冲×4B + reduceTmp 4B = 28B/elem；
    // fp16 路径 3 队列×双缓冲×2B + 3 个 fp32 计算 buf×4B + reduceTmp 4B 同为 28B/elem
    int64_t ubAvail = ubSize_ - RESERVED_UB;
    int64_t perElemBytes = (QUEUE_NUM * DOUBLE_BUFFER + 1) * FLOAT_BYTES; // 7 buf × 4B
    int64_t ubTileSize = (ubAvail / perElemBytes) / VL * VL;              // FloorAlign to VL
    OP_CHECK_IF(ubTileSize < VL, OP_LOGE(context_, "ub too small, ubTileSize=%ld", ubTileSize),
                return ge::GRAPH_FAILED);

    // 核数：每核至少 VL 个元素（1 个完整向量寄存器），不超过平台核数
    int64_t usedCores = totalN / VL;
    if (usedCores > coreNum_) {
        usedCores = coreNum_;
    }
    if (usedCores < 1) {
        usedCores = 1;
    }

    // 每核元素数（前 usedCores-1 核），尾核取余
    int64_t coreElements = totalN / usedCores;
    int64_t tailCoreElements = totalN - coreElements * (usedCores - 1);

    // 不使用 workspace（y2 归约走 y2 GM 原子加 / core0 串行，见 kernel 注释）
    int64_t wsBytes = 0;

    // 填 tiling data
    auto* tilingData = reinterpret_cast<FusedMulAddNL2lossTilingData*>(context_->GetRawTilingData()->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->totalElements = totalN;
    tilingData->coreElements = coreElements;
    tilingData->tailCoreElements = tailCoreElements;
    tilingData->ubTileSize = ubTileSize;
    context_->GetRawTilingData()->SetDataSize(sizeof(FusedMulAddNL2lossTilingData));

    context_->SetBlockDim(usedCores);
    context_->SetTilingKey(0); // 单一路径，key=0 对应 kernel 函数索引 _0
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = wsBytes;

    OP_LOGI(context_, "FusedMulAddNL2loss tiling: N=%ld, cores=%ld, coreElem=%ld, tailElem=%ld, ubTile=%ld, ws=%ld",
            totalN, usedCores, coreElements, tailCoreElements, ubTileSize, wsBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForFusedMulAddNL2loss(gert::TilingContext* context)
{
    FusedMulAddNL2lossTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForFusedMulAddNL2loss(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<FusedMulAddNL2lossCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedMulAddNL2loss)
    .Tiling(TilingForFusedMulAddNL2loss)
    .TilingParse<FusedMulAddNL2lossCompileInfo>(TilingPrepareForFusedMulAddNL2loss);

} // namespace optiling
