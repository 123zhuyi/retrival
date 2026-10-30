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
 * \file col2im_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/col2_im_tiling_data.h"
#include "../op_kernel/col2_im_tiling_key.h"

namespace optiling {

// 常量定义
const uint32_t BLOCK_SIZE = 32;
const uint32_t MIN_TILE_SIZE = 16;
const uint32_t BUFFER_NUM = 2;
const uint32_t UB_USAGE_THRESHOLD = 80; // UB使用率阈值 (%)
const uint32_t BITS_PER_VECTOR = 256;   // 向量位宽
const uint32_t BITS_PER_BYTE = 8;

// 属性索引常量
constexpr int32_t ATTR_OUTPUT_H = 0;
constexpr int32_t ATTR_OUTPUT_W = 1;
constexpr int32_t ATTR_KERNEL_H = 2;
constexpr int32_t ATTR_KERNEL_W = 3;
constexpr int32_t ATTR_STRIDE = 4;
constexpr int32_t ATTR_PADDING = 5;
constexpr int32_t ATTR_DILATION = 6;

// 默认卷积参数
constexpr int32_t DEFAULT_KERNEL_SIZE = 2;
constexpr int32_t DEFAULT_STRIDE = 1;
constexpr int32_t DEFAULT_PADDING = 0;
constexpr int32_t DEFAULT_DILATION = 1;

// Shape 维度索引
constexpr size_t DIM_N = 0;
constexpr size_t DIM_CHANNELS = 1;
constexpr size_t DIM_L = 2;
constexpr size_t EXPECTED_DIM_NUM = 3;

struct Col2ImCustomCompileInfo {};

// 内联工具函数
static inline uint32_t AlignUp(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

// 获取平台信息
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();

    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 获取输入Shape信息
static ge::graphStatus GetInputShapeInfo(gert::TilingContext* context, int64_t& N, int64_t& inputChannels, int64_t& L)
{
    auto inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto shape = Ops::Base::EnsureNotScalar(inputShape->GetStorageShape());
    auto dimNum = shape.GetDimNum();

    if (dimNum != EXPECTED_DIM_NUM) {
        return ge::GRAPH_FAILED;
    }

    N = shape.GetDim(DIM_N);
    inputChannels = shape.GetDim(DIM_CHANNELS);
    L = shape.GetDim(DIM_L);

    if (N <= 0 || inputChannels <= 0 || L <= 0) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 获取卷积属性参数
static ge::graphStatus GetConvAttrs(
    gert::TilingContext* context, int32_t& H, int32_t& W, int32_t& kernelH, int32_t& kernelW, int32_t& stride,
    int32_t& padding, int32_t& dilation)
{
    auto attrPtr = context->GetAttrs();
    if (attrPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 必需参数：输出尺寸
    if (attrPtr->GetInt(ATTR_OUTPUT_H)) {
        H = static_cast<int32_t>(*(attrPtr->GetInt(ATTR_OUTPUT_H)));
    } else {
        return ge::GRAPH_FAILED;
    }

    if (attrPtr->GetInt(ATTR_OUTPUT_W)) {
        W = static_cast<int32_t>(*(attrPtr->GetInt(ATTR_OUTPUT_W)));
    } else {
        return ge::GRAPH_FAILED;
    }

    // 可选参数：使用默认值
    kernelH =
        attrPtr->GetInt(ATTR_KERNEL_H) ? static_cast<int32_t>(*(attrPtr->GetInt(ATTR_KERNEL_H))) : DEFAULT_KERNEL_SIZE;
    kernelW =
        attrPtr->GetInt(ATTR_KERNEL_W) ? static_cast<int32_t>(*(attrPtr->GetInt(ATTR_KERNEL_W))) : DEFAULT_KERNEL_SIZE;
    stride = attrPtr->GetInt(ATTR_STRIDE) ? static_cast<int32_t>(*(attrPtr->GetInt(ATTR_STRIDE))) : DEFAULT_STRIDE;
    padding = attrPtr->GetInt(ATTR_PADDING) ? static_cast<int32_t>(*(attrPtr->GetInt(ATTR_PADDING))) : DEFAULT_PADDING;
    dilation =
        attrPtr->GetInt(ATTR_DILATION) ? static_cast<int32_t>(*(attrPtr->GetInt(ATTR_DILATION))) : DEFAULT_DILATION;

    if (H <= 0 || W <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || dilation <= 0) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 验证输入维度和计算输出通道数
static ge::graphStatus ValidateAndComputeChannels(int64_t inputChannels, int32_t kernelH, int32_t kernelW, int32_t& C)
{
    int32_t kernelSize = kernelH * kernelW;

    if (inputChannels % kernelSize != 0) {
        return ge::GRAPH_FAILED;
    }

    C = inputChannels / kernelSize;

    if (C <= 0) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 计算输出特征图尺寸
static void ComputeOutputFeatureSize(
    int32_t H, int32_t W, int32_t kernelH, int32_t kernelW, int32_t stride, int32_t padding, int32_t dilation,
    int32_t& outH, int32_t& outW)
{
    outH = (H + 2 * padding - dilation * (kernelH - 1) - 1) / stride + 1;
    outW = (W + 2 * padding - dilation * (kernelW - 1) - 1) / stride + 1;
}

// 获取数据类型大小
static ge::graphStatus GetDataTypeSize(gert::TilingContext* context, uint32_t& typeSize)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};

    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dataType = inputDesc->GetDataType();

    if (supportedDtype.count(dataType) == 0) {
        return ge::GRAPH_FAILED;
    }

    switch (dataType) {
        case ge::DT_FLOAT:
            typeSize = 4;
            break;
        case ge::DT_FLOAT16:
            typeSize = 2;
            break;
        default:
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// UB内存需求估算
static uint32_t EstimateUBUsage(int32_t pixelNum, int32_t kernelH, int32_t kernelW, uint32_t typeSize)
{
    uint32_t colDataPerPixel = kernelH * kernelW * typeSize;

    uint32_t outputBytes = AlignUp(pixelNum * typeSize, BLOCK_SIZE);
    uint32_t inputBytes = AlignUp(pixelNum * colDataPerPixel, BLOCK_SIZE);
    uint32_t tempBytes = AlignUp(pixelNum * typeSize, BLOCK_SIZE);

    return BUFFER_NUM * outputBytes + inputBytes + tempBytes;
}

// 计算核内tile大小
static ge::graphStatus CalculateIntraCoreTileSize(
    int64_t coreNum, uint64_t totalPixels, int32_t kernelH, int32_t kernelW, uint64_t ubSize, uint32_t typeSize,
    int32_t& tilePixelNum)
{
    int32_t basePixelsPerCore = totalPixels / coreNum;
    int32_t bigCoreNum = totalPixels % coreNum;
    int32_t maxPixelsPerCore = basePixelsPerCore + (bigCoreNum > 0 ? 1 : 0);

    tilePixelNum = MIN_TILE_SIZE;

    uint64_t ubThreshold = ubSize * UB_USAGE_THRESHOLD / 100;

    // 动态调整tile大小
    while (tilePixelNum * 2 <= maxPixelsPerCore &&
           EstimateUBUsage(tilePixelNum * 2, kernelH, kernelW, typeSize) <= ubThreshold) {
        tilePixelNum *= 2;
    }

    if (tilePixelNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 计算向量化对齐参数
static void ComputeVectorAlignment(
    uint32_t typeSize, int32_t tilePixelNum, uint32_t& vecElements, int32_t& alignedTileSize)
{
    vecElements = BITS_PER_VECTOR / (typeSize * BITS_PER_BYTE);
    alignedTileSize = ((tilePixelNum + vecElements - 1) / vecElements) * vecElements;
}

// 设置Tiling Key
static ge::graphStatus SetTilingKey(gert::TilingContext* context, ge::DataType dataType)
{
    uint64_t tilingKey = 0;

    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    } else {
        return ge::GRAPH_FAILED;
    }

    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// Workspace大小设置
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0; // Col2Im通常不需要额外workspace
    return ge::GRAPH_SUCCESS;
}

// Tiling主函数
static ge::graphStatus Col2ImCustomTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2. 获取输入Shape信息
    int64_t N, inputChannels, L;
    if (GetInputShapeInfo(context, N, inputChannels, L) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 3. 获取卷积属性参数
    int32_t H, W, kernelH, kernelW, stride, padding, dilation;
    if (GetConvAttrs(context, H, W, kernelH, kernelW, stride, padding, dilation) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 4. 验证维度并计算输出通道数
    int32_t C;
    if (ValidateAndComputeChannels(inputChannels, kernelH, kernelW, C) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 5. 计算输出特征图尺寸
    int32_t outH, outW;
    ComputeOutputFeatureSize(H, W, kernelH, kernelW, stride, padding, dilation, outH, outW);

    // 6. 获取数据类型大小
    uint32_t typeSize = 1;
    if (GetDataTypeSize(context, typeSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 7. 计算总像素数
    uint64_t inputElements = static_cast<uint64_t>(N) * inputChannels * L;
    uint64_t outputElements = static_cast<uint64_t>(N) * C * H * W;
    uint64_t totalPixels = outputElements;

    if (totalPixels == 0) {
        return ge::GRAPH_FAILED;
    }

    // 8. 计算核间划分参数
    int32_t basePixelsPerCore = totalPixels / coreNum;
    int32_t bigCoreNum = totalPixels % coreNum;

    // 9. 计算核内tile大小
    int32_t tilePixelNum;
    if (CalculateIntraCoreTileSize(coreNum, totalPixels, kernelH, kernelW, ubSize, typeSize, tilePixelNum) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 10. 计算向量化对齐参数
    uint32_t vecElements;
    int32_t alignedTileSize;
    ComputeVectorAlignment(typeSize, tilePixelNum, vecElements, alignedTileSize);

    // 11. 获取Workspace大小
    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 12. 初始化并填充tiling数据
    Col2ImTilingData* tiling = context->GetTilingData<Col2ImTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (memset_s(tiling, sizeof(Col2ImTilingData), 0, sizeof(Col2ImTilingData)) != EOK) {
        return ge::GRAPH_FAILED;
    }

    // 基础参数
    tiling->N = static_cast<int32_t>(N);
    tiling->C = C;
    tiling->H = H;
    tiling->W = W;
    tiling->kernel_h = kernelH;
    tiling->kernel_w = kernelW;
    tiling->stride_val = stride;
    tiling->padding_val = padding;
    tiling->dilation_val = dilation;
    tiling->input_channels = static_cast<int32_t>(inputChannels);
    tiling->L = static_cast<int32_t>(L);
    tiling->out_H = outH;
    tiling->out_W = outW;
    tiling->input_elements = inputElements;
    tiling->output_elements = outputElements;

    // 核间划分参数
    tiling->total_output_pixels = static_cast<uint32_t>(totalPixels);
    tiling->base_pixels_per_core = basePixelsPerCore;
    tiling->big_core_num = bigCoreNum;

    // 核内划分参数
    tiling->tile_pixel_num = tilePixelNum;
    tiling->aligned_tile_size = alignedTileSize;
    tiling->vec_elements = vecElements;

    // 13. 设置block维度
    context->SetBlockDim(coreNum);

    // 14. 设置Tiling Key
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dataType = inputDesc->GetDataType();
    if (SetTilingKey(context, dataType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 打印调试信息（可选）
    // printf("Col2Im Tiling: N=%d, C=%d, H=%d, W=%d, totalPixels=%u, coreNum=%ld, tilePixelNum=%d\n",
    //        tiling->N, tiling->C, tiling->H, tiling->W, tiling->total_output_pixels, coreNum, tilePixelNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCol2ImCustom([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// Tiling注册
IMPL_OP_OPTILING(Col2ImCustom)
    .Tiling(Col2ImCustomTilingFunc)
    .TilingParse<Col2ImCustomCompileInfo>(TilingParseForCol2ImCustom);

} // namespace optiling