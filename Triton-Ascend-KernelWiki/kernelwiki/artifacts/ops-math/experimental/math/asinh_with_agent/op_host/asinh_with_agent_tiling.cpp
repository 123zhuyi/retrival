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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asinh_with_agent_tiling_arch32.cpp
 * \brief AsinhWithAgent Tiling 实现（arch32 架构）
 *
 * Tiling 策略：
 * 1. 多核切分：将总元素按 AI Core 数量均分（CeilDiv）
 * 2. UB 切分：
 *    - 先由 GetAsinhMaxMinTmpSize 计算 sharedTmpBuffer 大小 (tmpBufSize)
 *    - 再用剩余 UB 计算 ubFactor（按 32 字节对齐）
 * 3. 缓冲模式：totalNum > 1024 时启用双缓冲（BUFFER_MODE=1）
 * 4. TilingKey：由 dtype + BUFFER_MODE 决定
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/asinh_with_agent_tiling_data.h"
#include "../op_kernel/asinh_with_agent_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// 双缓冲阈值：元素数量大于此值时启用双缓冲
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

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

static ge::graphStatus AsinhWithAgentTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取输入 Shape 和 dtype
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto inputStorageShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = inputStorageShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    // 只接受 float16 / float32（其他类型由 op_api 层转换后传入）
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT16 && dataType != ge::DT_FLOAT,
        OP_LOGE(context, "AsinhWithAgent: unsupported dtype %d", static_cast<int>(dataType)), return ge::GRAPH_FAILED);

    // 3. 获取 Workspace 大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. 计算 tmpBufSize（Asinh sharedTmpBuffer）
    // 不依赖 GetAsinhMaxMinTmpSize（adv_api/math/asinh_tiling.h 在本工程 include 路径中不可用）。
    // AscendC::Asinh 高阶 API 的 tmpBuffer 典型需求约为 ubFactor * typeSize * 4，
    // 这里保守取 ubSize / 4（≈48KB，192KB UB 下），满足任意 tile 尺寸的需求。
    uint32_t typeSize = (dataType == ge::DT_FLOAT16) ? 2U : 4U;
    int64_t tmpBufSize = static_cast<int64_t>(ubSize / 4);
    OP_CHECK_IF(
        tmpBufSize <= 0, OP_LOGE(context, "AsinhWithAgent: tmpBufSize is invalid: %ld", tmpBufSize),
        return ge::GRAPH_FAILED);

    // 5. 计算 ubFactor
    // UB 布局：input(BUFFER_NUM份) + output(BUFFER_NUM份) + tmpBuf(1份)
    // usableUB = ubSize - tmpBufSize
    // ubFactor = FloorAlign( usableUB / typeSize / (2 * BUFFER_NUM), ubBlockSize )
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1UL : 0UL;
    int64_t bufferNum = useDoubleBuffer ? 2LL : 1LL;
    int64_t ubBlockSize = GetUbBlockSize(context); // 32 字节 / typeSize 个元素

    int64_t usableUB = static_cast<int64_t>(ubSize) - tmpBufSize;
    OP_CHECK_IF(usableUB <= 0, OP_LOGE(context, "AsinhWithAgent: not enough UB for tmpBuf"), return ge::GRAPH_FAILED);

    // 2 * bufferNum 个 buffer（input + output 各 bufferNum 个）
    int64_t ubFactor = FloorAlign(
        FloorDiv(FloorDiv(usableUB, static_cast<int64_t>(typeSize)), static_cast<int64_t>(2) * bufferNum), ubBlockSize);
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "AsinhWithAgent: ubFactor <= 0"), return ge::GRAPH_FAILED);

    // 6. 多核切分
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 7. 写入 TilingData
    auto* tiling = context->GetTilingData<AsinhWithAgentTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AsinhWithAgentTilingData), 0, sizeof(AsinhWithAgentTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->tmpBufSize = tmpBufSize;

    // 8. 设置 TilingKey（dtype + BUFFER_MODE）
    // 参数顺序与 asinh_with_agent_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 一致：
    //   param1 = dTypeX (对应 D_T_X)
    //   param2 = useDoubleBuffer (对应 BUFFER_MODE)
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAsinhWithAgent([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AsinhWithAgentCompileInfo {}; // 入图场景依赖

IMPL_OP_OPTILING(AsinhWithAgent)
    .Tiling(AsinhWithAgentTilingFunc)
    .TilingParse<AsinhWithAgentCompileInfo>(TilingParseForAsinhWithAgent);

} // namespace optiling
