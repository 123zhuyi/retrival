/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
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
 * \file pad_v2_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/pad_v2_tiling_data.h"
#include "../op_kernel/pad_v2_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t WS_SYS_SIZE = 512U;
constexpr uint32_t RESERVED_BYTES = 512U;
static inline uint64_t AlignUp(uint64_t x, uint64_t a)
{
    if (a == 0) {
        return x;
    }
    return (x + a - 1) / a * a;
}

struct PadV2CompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
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
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    totalIdx = inputX->GetStorageShape().GetShapeSize();
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT16, ge::DT_INT32};
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
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
// 可直接替换你的 PadV2TilingFunc 内部实现（保留函数签名）
static ge::graphStatus PadV2TilingFunc(gert::TilingContext* context)
{
    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. shapes & dtype
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // handle empty input
    if (totalIdx <= 0) {
        PadV2TilingData* tiling = context->GetTilingData<PadV2TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(PadV2TilingData), 0, sizeof(PadV2TilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. tiling data
    PadV2TilingData* tiling = context->GetTilingData<PadV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(PadV2TilingData), 0, sizeof(PadV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // --- safer numeric types ---
    const auto xShape = context->GetInputTensor(0)->GetOriginShape();
    const auto inputDataType = context->GetInputTensor(0)->GetDataType();
    context->SetBlockDim(BLOCK_DIM);
    auto attrs = context->GetAttrs();
    int32_t mode = 0; // 默认值
    if (attrs) {
        const int64_t* attrA = attrs->GetInt(0);
        if (attrA != nullptr) {
            mode = *attrA;
        }
    }
    int32_t value = 0; // 默认值
    if (attrs) {
        const int64_t* attrA = attrs->GetInt(1);
        if (attrA != nullptr) {
            value = *attrA;
        }
    }
    int32_t pad[8] = {0};
    for (int i = 0; i < 8; i++) {
        if (attrs) {
            const int64_t* attrA = attrs->GetInt(i + 2);
            if (attrA != nullptr) {
                pad[i] = *attrA;
            }
        }
    }

    uint32_t typeSize = sizeof(float);
    switch (inputDataType) {
        case ge::DT_FLOAT16:
            typeSize = sizeof(uint16_t);
            break;
        case ge::DT_FLOAT:
            typeSize = sizeof(float);
            break;
        case ge::DT_INT32:
            typeSize = sizeof(int32_t);
            break;
        case ge::DT_INT16:
            typeSize = sizeof(int16_t);
            break;
        default:
            // 不支持的数据类型
            return ge::GRAPH_FAILED;
    }
    // 第一个输入 ->得到输入的形状 -> 每个维度相乘
    uint32_t totalLengthx = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    int32_t dimNum = xShape.GetDimNum();
    std::vector<int> dimarr(dimNum, 0);

    // 四维存储
    for (int i = 0; i < dimNum; i++) { // x的每个维度的数量
        dimarr[i] = xShape.GetDim(i);
    }
    // z的每个维度的数量
    std::vector<int> dimarrz(dimNum, 0);
    for (int i = 0; i < dimNum; i++) {
        dimarrz[i] = dimarr[i] + pad[2 * i] + pad[2 * i + 1];
    }
    uint32_t rows = 1;
    for (int i = 0; i < dimNum - 1; i++) {
        rows = rows * dimarr[i];
    }
    uint32_t rowz = 1;
    for (int i = 0; i < dimNum - 1; i++) {
        rowz = rowz * dimarrz[i];
    }
    // 第一次求和的空间大小--除开选择的维度其余的相乘
    uint32_t sumspace = totalLengthx;
    // 核间划分
    uint32_t big_core_num = rowz % BLOCK_DIM;
    uint32_t small_core_num = BLOCK_DIM - big_core_num;
    uint32_t small_tile_length = rowz / BLOCK_DIM;   // 小核分到的行数
    uint32_t big_tile_length = rowz / BLOCK_DIM + 1; // 大核分到的行数

    // 核内划分
    int64_t core_tile_x1 = 1; // 对行维度进行划分
    // ub内的数据划分-- 考虑temp就是第一次循环需要求和的数量
    auto FitsUB = [&](int64_t tile_x1) -> bool {
        uint64_t xBytes = AlignUp((uint64_t)tile_x1 * dimarrz[dimNum - 1] * typeSize, BLOCK_SIZE);
        uint64_t temp = AlignUp((uint64_t)tile_x1 * dimarrz[dimNum - 1] * typeSize, BLOCK_SIZE);
        uint64_t total = BUFFER_NUM * (xBytes + temp);
        // 预留 5% 余量
        return total <= (ubSize * 95 / 100);
    };

    // 调整核内分块参数以适应UB
    while (FitsUB(core_tile_x1) && core_tile_x1 < big_tile_length) {
        core_tile_x1 *= 2;
    }
    if (core_tile_x1 != 1) { // 判断是否倍增了 回溯到倍增前的结果
        core_tile_x1 /= 2;
    }
    if (!FitsUB(core_tile_x1)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t small_tile_times = small_tile_length / core_tile_x1;
    uint32_t big_tile_times = big_tile_length / core_tile_x1;
    uint32_t small_tail_num = small_tile_length % core_tile_x1;
    uint32_t big_tail_num = big_tile_length % core_tile_x1;
    if (small_tail_num != 0) {
        small_tile_times++;
    } else {
        small_tail_num = core_tile_x1;
    }
    if (big_tail_num != 0) {
        big_tile_times++;
    } else {
        big_tail_num = core_tile_x1;
    }

    // 填充的偏移量计算
    std::vector<int32_t> bias(dimNum - 1, 0);
    std::vector<int32_t> orign_bias(dimNum - 1, 0);

    // 初始化
    for (int i = 0; i < dimNum - 1; ++i) {
        bias[i] = 1;
        orign_bias[i] = 1;
    }

    int32_t acc_pad = 1;
    int32_t acc_orig = 1;
    // 计算 bias
    for (int i = dimNum - 2; i >= 0; --i) {
        bias[i] *= acc_pad;
        orign_bias[i] *= acc_orig;
        acc_pad *= dimarrz[i];
        acc_orig *= dimarr[i];
    }

    int32_t lpad = pad[(dimNum - 1) * 2];
    int32_t rpad = pad[(dimNum - 1) * 2 + 1];
    int32_t xlastdim = dimarr[dimNum - 1];

    tiling->small_tile_times = small_tile_times;
    tiling->big_tile_times = big_tile_times;
    tiling->small_tail_num = small_tail_num;
    tiling->big_tail_num = big_tail_num;
    tiling->totalLengthx = totalLengthx;
    tiling->big_core_num = big_core_num;
    tiling->small_core_num = small_core_num;
    tiling->small_tile_length = small_tile_length;
    tiling->big_tile_length = big_tile_length;
    tiling->core_tile_x1 = core_tile_x1;
    tiling->dimNum = dimNum;
    tiling->mode = mode;

    for (int i = 0; i < dimNum; ++i) {
        tiling->dimarr[i] = dimarr[i];
        tiling->dimarrz[i] = dimarrz[i];
        tiling->bias[i] = bias[i];
        tiling->orign_bias[i] = orign_bias[i];
    }
    for (uint32_t i = 0; i < BLOCK_DIM; ++i) {
        tiling->pad[i] = pad[i];
    }
    tiling->sumspace = sumspace;
    tiling->rowz = rowz;
    tiling->lpad = lpad;
    tiling->rpad = rpad;
    tiling->value = value;
    tiling->xlastdim = xlastdim;
    return ge::GRAPH_SUCCESS;
}
// tiling注册入口.
IMPL_OP_OPTILING(PadV2).Tiling(PadV2TilingFunc);
} // namespace optiling