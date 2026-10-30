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
 * \file acosh_tiling.cpp
 * \brief Acosh 算子 Tiling 实现（arch35）
 *
 * 切分策略：
 *   - 多核切分：按元素总数均分到各 AI Core
 *   - UB 切分：按可用 UB 大小和缓冲区数量均分
 *   - TilingKey：dtype × buffer_mode 共 6 个组合（A~F）
 *
 * TilingKey 模板参数（ASCENDC_TPL_SEL_PARAM 传参顺序与 acosh_tiling_key.h 中
 * ASCENDC_TPL_ARGS_DECL 定义一致）：
 *   参数1: dTypeX    (uint32_t) — 枚举值对应 C_DT_FLOAT16 / C_DT_FLOAT / C_DT_BF16
 *   参数2: useDoubleBuffer (uint64_t) — 0=单缓冲, 1=双缓冲
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/acosh_tiling_data.h"
#include "../../op_kernel/arch35/acosh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

// Elementwise 算子无需系统 workspace
constexpr uint32_t WS_SYS_SIZE = 0U;

// 双缓冲阈值：总元素数大于此值时启用双缓冲
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

// 标量输入转换为 {1}，保持后续逻辑统一
static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

// 获取平台信息：ubSize、coreNum
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

// 获取输入 shape 和 dtype 信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    totalNum = inputShapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Acosh: unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// Tiling 分发入口
static ge::graphStatus AcoshTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取 shape/dtype 信息
    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. 空 Tensor 快速返回
    if (totalNum == 0) {
        AcoshTilingData* tiling = context->GetTilingData<AcoshTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(
            memset_s(tiling, sizeof(AcoshTilingData), 0, sizeof(AcoshTilingData)) != EOK,
            OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        uint64_t useDoubleBuffer = 0;
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
        SetWorkspaceSize(context);
        return ge::GRAPH_SUCCESS;
    }

    // 4. 设置 workspace
    OP_CHECK_IF(
        SetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 5. 填充 TilingData
    AcoshTilingData* tiling = context->GetTilingData<AcoshTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AcoshTilingData), 0, sizeof(AcoshTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 多核切分
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // UB 切分
    // 确定缓冲模式
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;

    // 按 dtype 计算元素占用字节数
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2; // fp32=4B; fp16/bf16=2B

    // fp16/bf16 路径均走 Cast 回退（fp16/bf16 -> fp32 -> Acosh -> fp32 -> fp16/bf16）
    // 需要 fp32 中转缓冲，保守方案按 3 buffer（单缓冲）或 6 buffer（双缓冲）计算
    // fp32 路径：单缓冲 2 buffer，双缓冲 4 buffer（inQueue + outQueue 各 1/2）
    int64_t bufferNum;
    if (dataType == ge::DT_FLOAT) {
        bufferNum = useDoubleBuffer ? 4 : 2;
        tiling->ubFactor = FloorAlign(
            FloorDiv(static_cast<int64_t>(ubSize / typeSize), bufferNum),
            static_cast<int64_t>(GetUbBlockSize(context)));
    } else {
        // fp16 和 bf16 均走 Cast 路径，ubFactor 基于 fp32 字节数计算，确保中转缓冲能放下
        bufferNum = useDoubleBuffer ? 6 : 3;
        tiling->ubFactor = FloorAlign(
            FloorDiv(static_cast<int64_t>(ubSize / 4), bufferNum), static_cast<int64_t>(GetUbBlockSize(context)));
    }

    // ubFactor 不能为 0
    OP_CHECK_IF(tiling->ubFactor <= 0, OP_LOGE(context, "ubFactor is 0"), return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);

    // 6. 设置 TilingKey
    // 参数顺序与 acosh_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 一致：dTypeX, useDoubleBuffer
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAcosh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AcoshCompileInfo {};

IMPL_OP_OPTILING(Acosh).Tiling(AcoshTilingFunc).TilingParse<AcoshCompileInfo>(TilingParseForAcosh);

} // namespace optiling
