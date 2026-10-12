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
 * \file tabulate_fusion_grad_tiling.cpp
 * \brief Tiling implementation for tabulate_fusion_grad operator
 *
 * 多核切分: 通用两步法 (物理核均分 -> 反推实际核数) + PER_CORE_MIN_LOC=1024 下限保护.
 * 工作单元 = loc, Grid-Stride 遍历.
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../../op_kernel/arch35/tabulate_fusion_grad_tiling_data.h"
#include "../../op_kernel/arch35/tabulate_fusion_grad_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

using Ops::Base::EnsureNotScalar;

constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALIGN_64 = 64;
constexpr int64_t PER_CORE_MIN_LOC = 1024; // 下限保护, 对齐 32

struct TabulateFusionGradCompileInfo {};

// ============================================================================
// GetPlatformInfo: ubSize and AIV core num
// ============================================================================

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// GetShapeAttrsInfo: extract nloc/nnei/lastLayerSize/sizeAlign64/tableDim0
//   em: (nloc, nnei, 4)         -> index 3
//   descriptor: (nloc, 4, L)    -> index 5
//   table: (N_table, 6*align64) -> index 0
// ============================================================================

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& nloc, int64_t& nnei,
                                         int64_t& lastLayerSize, int64_t& sizeAlign64, int64_t& tableDim0)
{
    // em: [nloc, nnei, 4]
    auto emInput = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, emInput);
    auto emShape = EnsureNotScalar(emInput->GetStorageShape());
    OP_CHECK_IF(emShape.GetDimNum() != 3,
                OP_LOGE(context, "TabulateFusionGrad: em should be 3D, got %zu", emShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(emShape.GetDim(2) != 4,
                OP_LOGE(context, "TabulateFusionGrad: em.shape[2] should be 4, got %ld", emShape.GetDim(2)),
                return ge::GRAPH_FAILED);
    nloc = emShape.GetDim(0);
    nnei = emShape.GetDim(1);

    // descriptor: [nloc, 4, L] -> lastLayerSize = shape[2]
    auto descInput = context->GetInputShape(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, descInput);
    auto descShape = EnsureNotScalar(descInput->GetStorageShape());
    OP_CHECK_IF(descShape.GetDimNum() != 3,
                OP_LOGE(context, "TabulateFusionGrad: descriptor should be 3D, got %zu", descShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(descShape.GetDim(1) != 4,
                OP_LOGE(context, "TabulateFusionGrad: descriptor.shape[1] should be 4, got %ld", descShape.GetDim(1)),
                return ge::GRAPH_FAILED);
    lastLayerSize = descShape.GetDim(2);
    OP_CHECK_IF(lastLayerSize <= 0,
                OP_LOGE(context, "TabulateFusionGrad: last_layer_size should be > 0, got %ld", lastLayerSize),
                return ge::GRAPH_FAILED);

    // dy: [nloc, 4, L] -> shape[2] should == descriptor.shape[2]
    auto dyInput = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyInput);
    auto dyShape = EnsureNotScalar(dyInput->GetStorageShape());
    OP_CHECK_IF(dyShape.GetDimNum() != 3,
                OP_LOGE(context, "TabulateFusionGrad: dy should be 3D, got %zu", dyShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dyShape.GetDim(1) != 4,
                OP_LOGE(context, "TabulateFusionGrad: dy.shape[1] should be 4, got %ld", dyShape.GetDim(1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dyShape.GetDim(2) != lastLayerSize,
                OP_LOGE(context, "TabulateFusionGrad: dy.shape[2] %ld != descriptor.shape[2] %ld", dyShape.GetDim(2),
                        lastLayerSize),
                return ge::GRAPH_FAILED);

    // sizeAlign64
    sizeAlign64 = ((lastLayerSize + ALIGN_64 - 1) / ALIGN_64) * ALIGN_64;

    // table: [N_table, 6*sizeAlign64]
    auto tableInput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableInput);
    auto tableShape = EnsureNotScalar(tableInput->GetStorageShape());
    OP_CHECK_IF(tableShape.GetDimNum() != 2,
                OP_LOGE(context, "TabulateFusionGrad: table should be 2D, got %zu", tableShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tableShape.GetDim(1) != sizeAlign64 * 6,
                OP_LOGE(context, "TabulateFusionGrad: table.shape[1] %ld != sizeAlign64*6 %ld", tableShape.GetDim(1),
                        sizeAlign64 * 6),
                return ge::GRAPH_FAILED);
    tableDim0 = tableShape.GetDim(0);

    // table_info: size should >= 5
    auto tableInfoInput = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableInfoInput);
    auto tableInfoShape = EnsureNotScalar(tableInfoInput->GetStorageShape());
    OP_CHECK_IF(tableInfoShape.GetShapeSize() < 5,
                OP_LOGE(context, "TabulateFusionGrad: size of table_info should be >= 5, got %ld",
                        tableInfoShape.GetShapeSize()),
                return ge::GRAPH_FAILED);

    // em_x: total elements should == nloc * nnei
    auto emXInput = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, emXInput);
    auto emXShape = EnsureNotScalar(emXInput->GetStorageShape());
    OP_CHECK_IF(emXShape.GetShapeSize() != nloc * nnei,
                OP_LOGE(context, "TabulateFusionGrad: em_x total %ld != nloc*nnei %ld*%ld", emXShape.GetShapeSize(),
                        nloc, nnei),
                return ge::GRAPH_FAILED);

    // dtype check: only float32 supported
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT};
    for (size_t i = 0; i < 6; i++) {
        auto desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        ge::DataType dt = desc->GetDataType();
        OP_CHECK_IF(supportedDtype.count(dt) == 0,
                    OP_LOGE(context, "TabulateFusionGrad: unsupported dtype at input %zu (only float32)", i),
                    return ge::GRAPH_FAILED);
    }

    // nloc == 0 (空 tensor): 允许, needCoreNum 将为 0
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// GetWorkspaceSize: no user workspace, but must include system workspace
// ============================================================================

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    int64_t userWorkspaceSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Main tiling entry
// ============================================================================

static ge::graphStatus TabulateFusionGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    (void)ubSize; // MDE §5.5: SetLocalMemorySize(0)，ubSize 不再用于 DCache 计算

    int64_t nloc = 0, nnei = 0, lastLayerSize = 0, sizeAlign64 = 0, tableDim0 = 0;
    OP_CHECK_IF(GetShapeAttrsInfo(context, nloc, nnei, lastLayerSize, sizeAlign64, tableDim0) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 多核切分: 通用两步法 + PER_CORE_MIN_LOC 下限保护
    int64_t perCoreNloc = Ops::Base::CeilDiv(nloc, coreNum);
    if (perCoreNloc < PER_CORE_MIN_LOC) {
        perCoreNloc = ((PER_CORE_MIN_LOC + ALIGN_32 - 1) / ALIGN_32) * ALIGN_32;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(nloc, perCoreNloc);
    // MDE §2.1: nloc=0 时 needCoreNum=0 (语义: 无计算工作).
    // ascend950 运行时拒绝 block_dim=0 (INVALID_TILING), 故 block_dim 至少为 1;
    // kernel 的 grid-stride 循环在 nloc=0 时自然不执行 (locEnd = min(locStart+perCoreNloc, 0) = 0).

    // split_count=1 主流程 (split_count=2 未实现, 默认 1)
    int32_t splitCount = 1;
    int32_t splitIndex = 0;
    int32_t locStartOffset = 0;

    // 填充 TilingData
    TabulateFusionGradTilingData* tiling = context->GetTilingData<TabulateFusionGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TabulateFusionGradTilingData), 0, sizeof(TabulateFusionGradTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
    tiling->perCoreNloc = static_cast<int32_t>(perCoreNloc);
    tiling->nloc = static_cast<int32_t>(nloc);
    tiling->nnei = static_cast<int32_t>(nnei);
    tiling->lastLayerSize = static_cast<int32_t>(lastLayerSize);
    tiling->sizeAlign64 = static_cast<int32_t>(sizeAlign64);
    tiling->tableDim0 = static_cast<int32_t>(tableDim0);
    tiling->locStartOffset = locStartOffset;
    tiling->splitCount = splitCount;
    tiling->splitIndex = splitIndex;

    // ascend950 要求 block_dim >= 1; nloc=0 时启动 1 核, kernel grid-stride 因 nloc=0 自然不执行
    int64_t blockDim = (needCoreNum > 0) ? needCoreNum : 1;
    context->SetBlockDim(blockDim);
    // MDE §5.5: 主流程不使用 UB，SetLocalMemorySize(0) 使 DCache 取最大值 (248KB)
    auto res = context->SetLocalMemorySize(0);
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetLocalMemorySize(0) failed"), return ge::GRAPH_FAILED);

    // 单场景模式, dtype 由 DTYPE_TABLE 宏实例化
    uint64_t tilingKey = GET_TPL_TILING_KEY(TABULATE_FUSION_GRAD_MODE_DEFAULT);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// TilingParse callback (required by R15 three-part registration)
// ============================================================================

static ge::graphStatus TilingParseForTabulateFusionGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TabulateFusionGrad)
    .Tiling(TabulateFusionGradTilingFunc)
    .TilingParse<TabulateFusionGradCompileInfo>(TilingParseForTabulateFusionGrad);
} // namespace optiling
