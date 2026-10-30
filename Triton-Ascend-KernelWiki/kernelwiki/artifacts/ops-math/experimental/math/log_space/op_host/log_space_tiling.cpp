/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file log_space_tiling.cpp
 * \brief LogSpace Tiling 实现（arch35 / ascend950）
 *
 * 完整实现 6 个 TilingKey（fp32/fp16/bf16 × NORMAL/SINGLE）。
 * Tiling 按输出 dtype 计算 stepF/logBase/分核策略，TilingKey 通过 (D_T_Y, MODE) 二元组下发。
 */

#include <cmath>
#include <climits>
#include <limits>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/log_space_tiling_data.h"
#include "../op_kernel/log_space_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_PER_CORE = 64;     // 每个核最少处理元素数，小于该值则缩减核数
constexpr int64_t UB_CHUNK_ELEMS = 2048; // 单次搬运粒度（fp32 元素数）

// 属性索引常量（与 op_host/log_space_def.cpp / op_host/log_space_infershape.cpp 对齐）
constexpr int ATTR_IDX_START = 0;
constexpr int ATTR_IDX_END = 1;
constexpr int ATTR_IDX_STEPS = 2;
constexpr int ATTR_IDX_BASE = 3;

// MODE 枚举（必须与 log_space_tiling_key.h 一致）
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SINGLE = 1;

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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LogSpaceTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 属性
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* startPtr = attrs->GetAttrPointer<float>(ATTR_IDX_START);
    const float* endPtr = attrs->GetAttrPointer<float>(ATTR_IDX_END);
    const int64_t* stepsPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_STEPS);
    const float* basePtr = attrs->GetAttrPointer<float>(ATTR_IDX_BASE);
    OP_CHECK_NULL_WITH_CONTEXT(context, startPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, endPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, stepsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, basePtr);

    const float startF = *startPtr;
    const float endF = *endPtr;
    const int64_t steps = *stepsPtr;
    const float baseF = *basePtr;

    OP_CHECK_IF(steps < 0, OP_LOGE(context, "steps must be >= 0, got %ld", steps), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        steps > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        OP_LOGE(context, "steps must be <= UINT32_MAX (%u), got %ld", std::numeric_limits<uint32_t>::max(), steps),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseF <= 0.0f, OP_LOGE(context, "base must be > 0, got %f", baseF), return ge::GRAPH_FAILED);

    // 3. 输出 dtype
    auto outDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    ge::DataType dtype = outDesc->GetDataType();
    OP_CHECK_IF(
        dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
        OP_LOGE(context, "unsupported dtype %d", static_cast<int>(dtype)), return ge::GRAPH_FAILED);

    // 4. 工作空间
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 5. 填充 TilingData
    LogSpaceTilingData* tiling = context->GetTilingData<LogSpaceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LogSpaceTilingData), 0, sizeof(LogSpaceTilingData)) != EOK,
        OP_LOGE(context, "memset tiling data failed"), return ge::GRAPH_FAILED);

    tiling->totalLen = static_cast<uint64_t>(steps);
    tiling->startF = startF;
    tiling->logBase = std::log(baseF);
    tiling->ubChunk = static_cast<uint32_t>(UB_CHUNK_ELEMS);

    uint32_t mode = MODE_NORMAL;
    int64_t usedCoreNum = 1;
    if (steps <= 0) {
        // steps==0：空 tensor，Host 侧已短路（理论上不会走到这里），保险起见设置单核空跑
        tiling->coreNum = 1;
        tiling->tileLen = 0;
        tiling->tailTileLen = 0;
        tiling->tailCoreIdx = 0;
        tiling->stepF = 0.0f;
        mode = MODE_SINGLE;
    } else if (steps == 1) {
        tiling->coreNum = 1;
        tiling->tileLen = 1;
        tiling->tailTileLen = 1;
        tiling->tailCoreIdx = 0;
        tiling->stepF = 0.0f;
        mode = MODE_SINGLE;
        usedCoreNum = 1;
    } else {
        // steps >= 2: 常规多步生成
        int64_t maxCores = CeilDiv(steps, MIN_PER_CORE);
        int64_t useCores = (maxCores < coreNum) ? maxCores : coreNum;
        if (useCores < 1)
            useCores = 1;
        int64_t tileLen = steps / useCores;
        int64_t tailLen = steps - tileLen * (useCores - 1);

        tiling->coreNum = static_cast<uint32_t>(useCores);
        tiling->tileLen = static_cast<uint32_t>(tileLen);
        tiling->tailTileLen = static_cast<uint32_t>(tailLen);
        tiling->tailCoreIdx = static_cast<uint32_t>(useCores - 1);
        tiling->stepF = (endF - startF) / static_cast<float>(steps - 1);
        mode = MODE_NORMAL;
        usedCoreNum = useCores;
    }

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 6. TilingKey 选择：D_T_Y × MODE
    uint32_t dTypeY = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dTypeY, mode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLogSpace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LogSpaceCompileInfo {};

IMPL_OP_OPTILING(LogSpace).Tiling(LogSpaceTilingFunc).TilingParse<LogSpaceCompileInfo>(TilingParseForLogSpace);

} // namespace optiling
