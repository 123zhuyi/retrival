/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
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
 * \file im2_col_tiling.cpp
 * \brief Im2Col Custom operator tiling implementation with dynamic core allocation
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/im2_col_tiling_data.h"
#include "../op_kernel/im2_col_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const int32_t MAX_USE_CORE_NUM = 32;
const uint32_t MIN_TILE_SIZE = 16;
const uint32_t BUFFER_NUM = 2;
const uint32_t VEC_LEN = 8;
const uint32_t WS_SYS_SIZE = 512U;
constexpr int32_t ATTRPOS0 = 0;
constexpr int32_t ATTRPOS1 = 1;
constexpr int32_t ATTRPOS2 = 2;
constexpr int32_t ATTRPOS3 = 3;
constexpr int32_t ATTRPOS4 = 4;
constexpr int32_t ATTRPOS5 = 5;
constexpr int32_t ATTRPOS6 = 6;
constexpr int32_t ATTRPOS7 = 7;

struct Im2ColCustomCompileInfo {};

static uint32_t AlignUp(uint32_t a, uint32_t b)
{
    if (b == 0)
        return a;
    return (a + b - 1) / b * b;
}

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();

    // 限制核数不超过最大值
    if (coreNum > MAX_USE_CORE_NUM) {
        coreNum = MAX_USE_CORE_NUM;
    }

    // 检查coreNum是否有效
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "Invalid coreNum: %ld", coreNum), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 检查ubSize是否有效
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "Invalid ubSize: %lu", ubSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 设置基础参数到tiling数据结构
static ge::graphStatus SetBasicTilingParams(
    gert::TilingContext* context, Im2ColTilingData* tiling, int64_t N, int64_t C, int64_t H, int64_t W,
    int32_t kernel_h, int32_t kernel_w, int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w,
    int32_t dilation_h, int32_t dilation_w, int32_t out_H, int32_t out_W, int32_t L, int32_t output_channels,
    int64_t inputElements, int64_t outputElements)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 设置基础参数
    tiling->N = N;
    tiling->C = C;
    tiling->H = H;
    tiling->W = W;
    tiling->kernel_h = kernel_h;
    tiling->kernel_w = kernel_w;
    tiling->stride_h = stride_h;
    tiling->stride_w = stride_w;
    tiling->pad_h = pad_h;
    tiling->pad_w = pad_w;
    tiling->dilation_h = dilation_h;
    tiling->dilation_w = dilation_w;
    tiling->out_H = out_H;
    tiling->out_W = out_W;
    tiling->L = L;
    tiling->output_channels = output_channels;
    tiling->input_elements = inputElements;
    tiling->output_elements = outputElements;
    tiling->total_output_elements = outputElements;

    return ge::GRAPH_SUCCESS;
}

// 获取形状和数据类型信息(支持dilation)
static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t& N, int64_t& C, int64_t& H, int64_t& W, int32_t& kernel_h, int32_t& kernel_w,
    int32_t& stride_h, int32_t& stride_w, int32_t& pad_h, int32_t& pad_w, int32_t& dilation_h, int32_t& dilation_w,
    ge::DataType& dataType, uint32_t& typeSize)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    // 如果输入shape 是标量 转换为{1},否则保持原 shape 不变
    auto inputShapeX = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());

    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);

    auto dimNum = inputShapeX.GetDimNum();

    // Im2Col只支持3D或4D输入
    OP_CHECK_IF(
        dimNum != 3 && dimNum != 4,
        OP_LOGE(context, "Im2ColCustom: only support 3D or 4D input, but got dimNum=%zu", dimNum),
        return ge::GRAPH_FAILED);

    // 根据维度数解析shape
    if (dimNum == 4) {
        // 4D: [N, C, H, W]
        N = inputShapeX.GetDim(0);
        C = inputShapeX.GetDim(1);
        H = inputShapeX.GetDim(2);
        W = inputShapeX.GetDim(3);
    } else if (dimNum == 3) {
        // 3D: [C, H, W] - 单batch
        N = 1;
        C = inputShapeX.GetDim(0);
        H = inputShapeX.GetDim(1);
        W = inputShapeX.GetDim(2);
    }

    OP_CHECK_IF(
        N <= 0 || C <= 0 || H <= 0 || W <= 0,
        OP_LOGE(context, "Im2ColCustom: invalid input shape N=%ld, C=%ld, H=%ld, W=%ld", N, C, H, W),
        return ge::GRAPH_FAILED);

    // 获取卷积参数(支持dilation)
    kernel_h = 2;   // 默认值
    kernel_w = 2;   // 默认值
    stride_h = 1;   // 默认值
    stride_w = 1;   // 默认值
    pad_h = 0;      // 默认值
    pad_w = 0;      // 默认值
    dilation_h = 1; // 默认值
    dilation_w = 1; // 默认值

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    // 获取属性值
    if (attrPtr->GetInt(ATTRPOS0)) {
        kernel_h = static_cast<int32_t>(*(attrPtr->GetInt(ATTRPOS0)));
    }
    if (attrPtr->GetInt(ATTRPOS1)) {
        kernel_w = static_cast<int32_t>(*(attrPtr->GetInt(ATTRPOS1)));
    }
    if (attrPtr->GetInt(ATTRPOS2)) {
        stride_h = static_cast<int32_t>(*(attrPtr->GetInt(ATTRPOS2)));
    }
    if (attrPtr->GetInt(ATTRPOS3)) {
        stride_w = static_cast<int32_t>(*(attrPtr->GetInt(ATTRPOS3)));
    }
    if (attrPtr->GetInt(ATTRPOS4)) {
        pad_h = static_cast<int32_t>(*(attrPtr->GetInt(ATTRPOS4)));
    }
    if (attrPtr->GetInt(ATTRPOS5)) {
        pad_w = static_cast<int32_t>(*(attrPtr->GetInt(ATTRPOS5)));
    }
    if (attrPtr->GetInt(ATTRPOS6)) {
        dilation_h = static_cast<int32_t>(*(attrPtr->GetInt(ATTRPOS6)));
    }
    if (attrPtr->GetInt(ATTRPOS7)) {
        dilation_w = static_cast<int32_t>(*(attrPtr->GetInt(ATTRPOS7)));
    }

    OP_CHECK_IF(
        kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0 || dilation_h <= 0 ||
            dilation_w <= 0,
        OP_LOGE(
            context, "Im2ColCustom: invalid params kernel=(%d,%d), stride=(%d,%d), pad=(%d,%d), dilation=(%d,%d)",
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w),
        return ge::GRAPH_FAILED);

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }

    // 获取数据类型
    switch (dataType) {
        case ge::DT_FLOAT:
            typeSize = 4;
            break;
        case ge::DT_FLOAT16:
            typeSize = 2;
            break;
        default:
            OP_LOGE(context, "Im2ColCustom: unsupported data type");
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 计算Im2Col输出尺寸(支持dilation)
static ge::graphStatus CalculateOutputDims(
    gert::TilingContext* context, int64_t H, int64_t W, int64_t C, int32_t kernel_h, int32_t kernel_w, int32_t stride_h,
    int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t dilation_h, int32_t dilation_w, int32_t& out_H,
    int32_t& out_W, int32_t& L, int32_t& output_channels)
{
    // 支持dilation的输出尺寸计算
    out_H = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    out_W = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    OP_CHECK_IF(
        out_H <= 0 || out_W <= 0,
        OP_LOGE(context, "Im2ColCustom: invalid output dims out_H=%d, out_W=%d", out_H, out_W),
        return ge::GRAPH_FAILED);

    L = out_H * out_W;
    output_channels = C * kernel_h * kernel_w;

    return ge::GRAPH_SUCCESS;
}

// UB内存需求估算
static uint32_t EstimateUBUsage(int32_t elem_num, uint32_t typeSize)
{
    // 输入数据缓存：需要缓存可能被重复访问的输入区域
    uint32_t inputBytes = AlignUp(elem_num * 4 * typeSize, BLOCK_SIZE); // 预估4倍输入数据
    // 输出数据
    uint32_t outputBytes = AlignUp(elem_num * typeSize, BLOCK_SIZE);
    // 临时数据：索引计算和工作缓存
    uint32_t workBytes = AlignUp(elem_num * sizeof(int32_t) + VEC_LEN * typeSize, BLOCK_SIZE);

    uint32_t total = BUFFER_NUM * (inputBytes + outputBytes) + workBytes;

    return total;
}

// 计算核内划分参数(无需核间数组分配)
static ge::graphStatus CalculateIntraCorePartition(
    int64_t totalOutputElements, int64_t coreNum, uint64_t ubSize, uint32_t typeSize, int32_t& baseElementsPerCore,
    int32_t& bigCoreNum, int32_t& tileElementNum)
{
    // 计算基础划分参数
    baseElementsPerCore = totalOutputElements / coreNum;
    bigCoreNum = totalOutputElements % coreNum;

    // 计算最大每个核处理的元素数
    int32_t maxElementsPerCore = baseElementsPerCore + (bigCoreNum > 0 ? 1 : 0);

    // 动态调整tile大小以充分利用UB内存
    tileElementNum = 1;
    while (tileElementNum * 2 <= maxElementsPerCore &&
           EstimateUBUsage(tileElementNum * 2, typeSize) <= ubSize * 90 / 100) {
        tileElementNum *= 2;
    }

    // 确保最小tile大小
    if ((uint32_t)tileElementNum < MIN_TILE_SIZE) {
        tileElementNum = std::min(MIN_TILE_SIZE, static_cast<uint32_t>(maxElementsPerCore));
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

// Tiling分发入口
static ge::graphStatus Im2ColCustomTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取shape、属性信息(支持dilation)
    int64_t N = 0, C = 0, H = 0, W = 0;
    int32_t kernel_h = 0, kernel_w = 0, stride_h = 0, stride_w = 0;
    int32_t pad_h = 0, pad_w = 0, dilation_h = 0, dilation_w = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint32_t typeSize = 0;
    OP_CHECK_IF(
        GetShapeAttrsInfo(
            context, N, C, H, W, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, dataType,
            typeSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. 计算输出尺寸(支持dilation)
    int32_t out_H, out_W, L, output_channels;
    OP_CHECK_IF(
        CalculateOutputDims(
            context, H, W, C, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, out_H,
            out_W, L, output_channels) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateOutputDims error"), return ge::GRAPH_FAILED);

    // 计算总元素数
    int64_t inputElements = N * C * H * W;
    int64_t outputElements = static_cast<int64_t>(N) * output_channels * L;

    OP_CHECK_IF(outputElements == 0, OP_LOGE(context, "Im2ColCustom: outputElements is 0"), return ge::GRAPH_FAILED);

    // 4. 获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 5. 初始化tiling数据
    Im2ColTilingData* tiling = context->GetTilingData<Im2ColTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(Im2ColTilingData), 0, sizeof(Im2ColTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 6. 设置基础参数
    OP_CHECK_IF(
        SetBasicTilingParams(
            context, tiling, N, C, H, W, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            out_H, out_W, L, output_channels, inputElements, outputElements) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetBasicTilingParams error"), return ge::GRAPH_FAILED);

    // 7. 计算核间和核内划分(无数组,只保留基础参数)
    int32_t baseElementsPerCore, bigCoreNum, tileElementNum;
    OP_CHECK_IF(
        CalculateIntraCorePartition(
            outputElements, coreNum, ubSize, typeSize, baseElementsPerCore, bigCoreNum, tileElementNum) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateIntraCorePartition error"), return ge::GRAPH_FAILED);

    tiling->base_elements_per_core = baseElementsPerCore;
    tiling->big_core_num = bigCoreNum;
    tiling->tile_element_num = tileElementNum;

    // 8. 计算对齐参数
    int32_t vec_align_size = 32 / typeSize;
    int32_t aligned_element_size = ((tileElementNum + vec_align_size - 1) / vec_align_size) * vec_align_size;
    tiling->aligned_element_size = aligned_element_size;

    context->SetBlockDim(coreNum);

    // 9. 设置tiling key（根据数据类型）
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "Unsupported data type for tiling key");
        return ge::GRAPH_FAILED;
    }

    // 打印调试信息
    OP_LOGI(context, "Im2ColCustom Tiling: N=%ld, C=%ld, H=%ld, W=%ld", N, C, H, W);
    OP_LOGI(
        context, "Im2ColCustom Tiling: kernel=(%dx%d), stride=(%d,%d), pad=(%d,%d), dilation=(%d,%d)", kernel_h,
        kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    OP_LOGI(
        context, "Im2ColCustom Tiling: out_H=%d, out_W=%d, L=%d, outputChannels=%d, outputElements=%ld", out_H, out_W,
        L, output_channels, outputElements);
    OP_LOGI(
        context, "Im2ColCustom Tiling: coreNum=%ld, baseElementsPerCore=%d, bigCoreNum=%d, tileElementNum=%d", coreNum,
        baseElementsPerCore, bigCoreNum, tileElementNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForIm2ColCustom([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// Tiling注册入口
IMPL_OP_OPTILING(Im2Col)
    .Tiling(Im2ColCustomTilingFunc)
    .TilingParse<Im2ColCustomCompileInfo>(TilingParseForIm2ColCustom);
} // namespace optiling