/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
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
 * \file slice_v3_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/slice_v3_tiling_data.h"
#include "../op_kernel/slice_v3_tiling_key.h"

namespace optiling {
constexpr uint32_t WS_SYS_SIZE = 512U;
constexpr uint32_t RESERVED_BYTES = 512U;
static void ComputeCumShape(const std::vector<int64_t>& shape, int64_t* cumShape)
{
    int n = shape.size();
    if (n == 0)
        return;

    cumShape[n - 1] = 1;
    for (int i = n - 2; i >= 0; i--) {
        cumShape[i] = cumShape[i + 1] * shape[i + 1];
    }
}

struct SliceV3CompileInfo {};
const uint32_t BLOCK_SIZE = 32;

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus GetShapeAttrsInfo(const gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    totalIdx = inputX->GetStorageShape().GetShapeSize();
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_INT16,
                                                   ge::DT_BF16,  ge::DT_INT8,  ge::DT_UINT8};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus SliceV3TilingFunc(gert::TilingContext* context)
{
    // ------------- 获取并初始化 TilingData -------------
    SliceV3TilingData* tiling = context->GetTilingData<SliceV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SliceV3TilingData), 0, sizeof(SliceV3TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // ------------- 获取平台信息：UB 大小、Core 个数 -------------
    uint64_t ubSize;
    uint32_t coreNum;
    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = plat.GetCoreNum();

    // ------------- workspace 校验 -------------
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // ------------- 输入信息：维度、dataType -------------
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    uint32_t dim = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    tiling->dim = dim;

    uint32_t typeLen = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLen);
    OP_CHECK_IF(typeLen == 0, OP_LOGE(context, "dtype len error"), return ge::GRAPH_FAILED);

    // ------------- begin / size Tensor -------------
    auto beginTensor = context->GetInputTensor(1);
    auto sizeTensor = context->GetInputTensor(2);

    const int64_t* beginPtr = beginTensor->GetData<int64_t>();
    const int64_t* sizePtr = sizeTensor->GetData<int64_t>();

    std::vector<int64_t> begin(dim);
    std::vector<int64_t> size(dim);

    for (uint32_t i = 0; i < dim; ++i) {
        begin[i] = beginPtr[i];
        size[i] = sizePtr[i];
        tiling->begin[i] = begin[i];
    }
    // ------------- 计算输入、输出 Shape cumShape -------------
    std::vector<int64_t> inputDims(dim), outputDims(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        inputDims[i] = context->GetInputShape(0)->GetStorageShape().GetDim(i);
        outputDims[i] = size[i];
    }

    ComputeCumShape(inputDims, tiling->inputCumShape);
    ComputeCumShape(outputDims, tiling->outputCumShape);

    // ------------- 计算 totalBlocks 和 lastDim -------------
    // totalBlocks: 除最后一个维度外，所有输出维度的乘积
    // 表示需要并行处理的"数据块"数量，每个数据块对应一个外层循环迭代
    int64_t totalBlocks = 1;
    for (uint32_t i = 0; i < dim - 1; ++i) {
        if (outputDims[i] <= 0) {
            totalBlocks = 0;
            break;
        }
        totalBlocks *= outputDims[i];
    }

    int64_t lastDim = (dim > 0 ? outputDims[dim - 1] : 0);

    // ------------- 核分配 -------------
    uint32_t useCoreNum = std::min((uint32_t)std::max<int64_t>(totalBlocks, 1), coreNum);
    if (totalBlocks == 0)
        useCoreNum = 1;

    if (useCoreNum == 0) {
        OP_LOGE(context, "useCoreNum cannot be zero");
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(useCoreNum);

    // ------------- UB 切 tile 大小 -------------
    uint64_t availUb = (ubSize * 9 / 10) / 2; // 可用 UB 空间：使用 90% 作为安全余量，并预留一半用于双缓冲，避免 UB 溢出
    uint32_t alignNum = BLOCK_SIZE / typeLen;
    if (alignNum == 0)
        alignNum = 1;

    uint32_t maxTile = static_cast<uint32_t>(availUb / typeLen);
    uint32_t alignedMaxTile = (maxTile >= alignNum) ? (maxTile & (~(alignNum - 1))) : maxTile;

    if (alignedMaxTile == 0)
        alignedMaxTile = alignNum;

    int64_t tileDataNum = 1;
    if (lastDim > 0) {
        tileDataNum = (lastDim <= alignedMaxTile) ? lastDim : alignedMaxTile;
        if (tileDataNum < alignNum)
            tileDataNum = alignNum;
    }

    // ------------- 写入 tiling data -------------
    tiling->outerLoopNum = totalBlocks / useCoreNum;
    tiling->tailCoreNum = totalBlocks % useCoreNum;
    tiling->tileDataNum = tileDataNum;
    tiling->lastDimSize = lastDim;

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParseForSliceV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(SliceV3).Tiling(SliceV3TilingFunc).TilingParse<SliceV3CompileInfo>(TilingParseForSliceV3);
} // namespace optiling
