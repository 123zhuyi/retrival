/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
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
 * \file split_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "../op_kernel/split_tiling_data.h"
#include "../op_kernel/split_tiling_key.h"
namespace optiling {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint64_t UB_DATA_NUMBER_DEFAULT = 4; // 结合kernel侧tbuf和tque数量设置
constexpr uint32_t INDICES_LIMIT = 10;
constexpr uint32_t DIM_LIMIT = 8;
struct SplitCompileInfo {};
struct SplitCompileInfoShapeInfo {
    uint64_t inputNum{0};
    uint32_t inputBytes{0};
    uint32_t tileBlockNum{0};
    uint32_t tileDataNum{0};
    uint32_t inputLengthAlign32{0};
    uint32_t smallCoreDataNum{0};
    uint32_t bigCoreDataNum{0};
    uint32_t smallTailDataNum{0};
    uint32_t bigTailDataNum{0};
    uint32_t finalSmallTileNum{0};
    uint32_t finalBigTileNum{0};
    uint32_t tailBlockNum{0};
    uint32_t blockSize{0};
    uint64_t dimNum1{0};

    int64_t axis{0};
    uint32_t shape[DIM_LIMIT]{0};
    uint32_t indices_or_sections[INDICES_LIMIT]{0};
    uint32_t indices_len{0};
    uint32_t splitLen[INDICES_LIMIT + 1]{0};
    uint32_t unit{0};
    bool isEven{1};
    uint32_t srcdim{0};
};
// 平台信息
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
// 工作空间
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, SplitCompileInfoShapeInfo& info)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 系统workspace大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 用户workspace大小（64B 对齐，每个范数一个slot）
    size_t usrSize = info.inputLengthAlign32; // 用户部分
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(context, "currentWorkspace is nullptr"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus JudgeEven(SplitCompileInfoShapeInfo& info)
{
    bool isEven = true;
    if (info.indices_len > 1) {
        isEven = false; // 多个索引，不是均分
    }
    info.isEven = isEven;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ProductExceptAxis(SplitCompileInfoShapeInfo& info)
{
    uint64_t unit = 1;
    uint64_t srcdim = static_cast<uint64_t>(info.dimNum1);
    if (srcdim > 0) {
        if (info.axis >= static_cast<int64_t>(srcdim)) {
            unit = 1; // invalid axis: 根据策略选择设置 unit=1 或报错
        } else {
            for (uint32_t i = 0; i < srcdim; ++i) {
                if (i == info.axis)
                    continue;
                unit *= info.shape[i];
            }
        }
    }
    info.unit = unit;
    info.srcdim = srcdim;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PreLen(SplitCompileInfoShapeInfo& info)
{
    uint64_t splitLen[INDICES_LIMIT + 1] = {0}; // 均分后长度存入第0位，否则依次存入
    if (info.isEven) {
        splitLen[0] = info.inputNum / info.indices_or_sections[0];
    } else {
        if (info.axis >= 0) {
            splitLen[0] = info.indices_or_sections[0] * info.unit;
            for (uint32_t i = 1; i < info.indices_len; ++i) {
                splitLen[i] = (info.indices_or_sections[i] - info.indices_or_sections[i - 1]) * info.unit;
            }
            splitLen[info.indices_len] =
                (info.shape[info.axis] - info.indices_or_sections[info.indices_len - 1]) * info.unit;
        } else {
            splitLen[0] = info.indices_or_sections[0];
            for (uint32_t i = 1; i < info.indices_len; ++i) {
                splitLen[i] = (info.indices_or_sections[i] - info.indices_or_sections[i - 1]);
            }
            splitLen[info.indices_len] = info.inputNum - info.indices_or_sections[info.indices_len - 1];
        }
    }
    for (uint32_t i = 0; i < INDICES_LIMIT + 1; i++) {
        info.splitLen[i] = splitLen[i];
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrs(gert::TilingContext* context, SplitCompileInfoShapeInfo& info)
{
    const gert::TypedContinuousVector<int64_t>* indices_list = nullptr;
    auto attrs = context->GetAttrs();
    if (attrs) {
        if (attrs->GetListInt(0)) {
            indices_list = context->GetAttrs()->GetListInt(0);
        }
        if (attrs->GetInt(1)) {
            info.axis = *(attrs->GetInt(1));
        }
    }
    if (indices_list == nullptr) {
        OP_LOGE(context, "indices_list is nullptr");
        return ge::GRAPH_FAILED;
    }
    const int64_t* indices_or_sections0 = indices_list->GetData();
    int64_t indices_size = static_cast<int64_t>(indices_list->GetSize());
    info.indices_len = indices_size;
    // 将indices_or_sections数组复制到tiling结构中
    uint32_t actual_size = std::min(indices_size, static_cast<int64_t>(10)); // 限制最大复制10个元素（避免数组越界）
    for (uint32_t i = 0; i < actual_size; ++i) {
        info.indices_or_sections[i] = indices_or_sections0[i];
    }
    return ge::GRAPH_SUCCESS;
}

// 形状属性
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, SplitCompileInfoShapeInfo& info)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    info.inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = info.inputNum * typeLength;
    if (info.inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    info.inputBytes = typeLength;
    info.blockSize = Ops::Base::GetUbBlockSize(context);

    info.tileBlockNum = (ubSize / BUFFER_NUM / info.blockSize) / UB_DATA_NUMBER_DEFAULT;
    if (info.inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    info.tileDataNum = (info.tileBlockNum * info.blockSize) / info.inputBytes;
    info.inputLengthAlign32 = (((inputLength + info.blockSize - 1) / info.blockSize) * info.blockSize);

    const gert::Shape x1ShapeObj = context->GetInputShape(0)->GetStorageShape();
    size_t dimNum1 = x1ShapeObj.GetDimNum();
    OP_CHECK_IF(dimNum1 > DIM_LIMIT, OP_LOGE(context, "dimNum1 exceed limit"), return ge::GRAPH_FAILED);
    info.dimNum1 = dimNum1;

    for (uint32_t i = 0; i < dimNum1; ++i) {
        info.shape[i] = static_cast<uint32_t>(x1ShapeObj.GetDim(i));
    }
    ge::graphStatus ret = GetAttrs(context, info);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrs error"), return ge::GRAPH_FAILED);
    ret = JudgeEven(info);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "JudgeEven error"), return ge::GRAPH_FAILED);
    ret = ProductExceptAxis(info);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "ProductExceptAxis error"), return ge::GRAPH_FAILED);
    ret = PreLen(info);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "PreLen error"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
// 分块信息
static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, int64_t coreNum, SplitCompileInfoShapeInfo& info)
{
    OP_CHECK_IF(
        0 == info.blockSize || 0 == coreNum || 0 == info.tileBlockNum || 0 == info.inputBytes,
        OP_LOGE(context, "input is error"), return ge::GRAPH_FAILED);

    uint64_t everyCoreInputBlockNum = info.inputLengthAlign32 / info.blockSize / coreNum;
    info.tailBlockNum = (info.inputLengthAlign32 / info.blockSize) % coreNum;
    info.smallCoreDataNum = everyCoreInputBlockNum * info.blockSize / info.inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalSmallTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    info.smallTailDataNum = info.smallCoreDataNum - (info.tileDataNum * smallTileNum);
    info.smallTailDataNum = info.smallTailDataNum == 0 ? info.tileDataNum : info.smallTailDataNum;

    everyCoreInputBlockNum += 1;
    info.bigCoreDataNum = everyCoreInputBlockNum * info.blockSize / info.inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalBigTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    info.bigTailDataNum = info.bigCoreDataNum - info.tileDataNum * bigTileNum;
    info.bigTailDataNum = info.bigTailDataNum == 0 ? info.tileDataNum : info.bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingSetCommonData(
    gert::TilingContext* context, SplitCompileInfoShapeInfo& shapeInfo, SplitTilingData* tiling)
{
    OP_CHECK_IF(
        context == nullptr || tiling == nullptr, OP_LOGE(context, "context or tilingData is nullptr"),
        return ge::GRAPH_FAILED);
    // 设置tiling公共数据
    // 设置tiling数据
    tiling->smallCoreDataNum = static_cast<uint32_t>(shapeInfo.smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint32_t>(shapeInfo.bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint32_t>(shapeInfo.tileDataNum);
    tiling->smallTailDataNum = static_cast<uint32_t>(shapeInfo.smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint32_t>(shapeInfo.bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(shapeInfo.finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint32_t>(shapeInfo.finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint32_t>(shapeInfo.tailBlockNum);
    tiling->blockSize = static_cast<uint32_t>(shapeInfo.blockSize);
    tiling->axis = static_cast<int64_t>(shapeInfo.axis);
    for (uint32_t i = 0; i < DIM_LIMIT; i++) {
        tiling->shape[i] = static_cast<uint32_t>(shapeInfo.shape[i]);
    }
    tiling->indices_len = static_cast<uint32_t>(shapeInfo.indices_len);
    for (uint32_t i = 0; i < INDICES_LIMIT; i++) {
        tiling->indices_or_sections[i] = static_cast<uint32_t>(shapeInfo.indices_or_sections[i]);
    }
    tiling->isEven = static_cast<bool>(shapeInfo.isEven);
    tiling->unit = static_cast<uint32_t>(shapeInfo.unit);
    for (uint32_t i = 0; i < INDICES_LIMIT + 1; i++) {
        tiling->splitLen[i] = shapeInfo.splitLen[i];
    }
    tiling->totalNums = static_cast<uint32_t>(shapeInfo.inputNum);
    tiling->srcdim = static_cast<uint32_t>(shapeInfo.srcdim);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus SplitTilingFunc(gert::TilingContext* context)
{
    SplitTilingData* tiling = context->GetTilingData<SplitTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SplitTilingData), 0, sizeof(SplitTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 获取输入数据信息
    SplitCompileInfoShapeInfo shapeInfo;
    ret = GetShapeAttrsInfo(context, ubSize, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 计算coreNum
    if (shapeInfo.tileDataNum >= shapeInfo.inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (static_cast<uint32_t>(coreNum) < shapeInfo.inputLengthAlign32 / shapeInfo.blockSize) ?
                      coreNum :
                      shapeInfo.inputLengthAlign32 / shapeInfo.blockSize;
    }
    // 计算每个core处理的数据块数
    ret = CalculateCoreBlockNums(context, coreNum, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);
    ret = TilingSetCommonData(context, shapeInfo, tiling);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "TilingSetCommonData error"), return ge::GRAPH_FAILED);
    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context, shapeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    // 设置tilingKey.
    uint32_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSplit([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
// tiling注册入口.
IMPL_OP_OPTILING(Split).Tiling(SplitTilingFunc).TilingParse<SplitCompileInfo>(TilingParseForSplit);
} // namespace optiling