/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pow2_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/pow2_tiling_data.h"
#include "../op_kernel/pow2_tiling_key.h"

namespace optiling {

constexpr uint32_t SELECT_NEED = 256; // 256B
constexpr uint32_t BLOCK_SIZE = SELECT_NEED;
constexpr uint32_t DEFAULT_UB_NUM = 10;
constexpr uint32_t INT8_UB_NUM = 24;
constexpr uint32_t DIMS_LIMIT = 10;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;
struct Pow2CompileInfo {};
struct Pow2ShapeInfo {
    uint64_t inputNum{0};
    uint64_t inputBytes{0};
    uint64_t tileBlockNum{0};
    uint64_t tileDataNum{0};
    uint64_t inputLengthAlign256{0};
    uint64_t smallCoreDataNum{0};
    uint64_t bigCoreDataNum{0};
    uint64_t smallTailDataNum{0};
    uint64_t bigTailDataNum{0};
    uint64_t finalSmallTileNum{0};
    uint64_t finalBigTileNum{0};
    uint64_t tailBlockNum{0};

    bool is_input0_scalar{1};
    bool is_input1_scalar{1};
    uint64_t yDim{0};
    bool isSameX1{1};
    bool isSameX2{1};
    uint64_t strideX1[DIMS_LIMIT]{0};
    uint64_t strideX2[DIMS_LIMIT]{0};
    uint64_t strideY[DIMS_LIMIT]{0};
    uint64_t X2TotalNum{0};
    uint64_t X1TotalNum{0};
};

static ge::graphStatus TilingParseForPow2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

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
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, Pow2ShapeInfo& info)
{
    OP_CHECK_IF(
        context == nullptr || context->GetOutputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    info.inputNum = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetOutputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = info.inputNum * typeLength;
    if (info.inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    info.inputBytes = inputLength / info.inputNum;
    auto dataType1 = context->GetInputDesc(0)->GetDataType();
    auto dataType2 = context->GetInputDesc(1)->GetDataType();
    auto dataType3 = context->GetOutputDesc(0)->GetDataType();
    uint64_t ubDataNumber = DEFAULT_UB_NUM;
    if (dataType1 == ge::DT_INT8 || dataType1 == ge::DT_UINT8 || dataType2 == ge::DT_INT8 ||
        dataType2 == ge::DT_UINT8 || dataType3 == ge::DT_INT8 || dataType3 == ge::DT_UINT8) {
        ubDataNumber = INT8_UB_NUM;
    }
    if ((dataType1 == ge::DT_INT8 && dataType2 == ge::DT_UINT8) ||
        (dataType2 == ge::DT_INT8 && dataType1 == ge::DT_UINT8) ||
        (dataType1 == ge::DT_INT8 && dataType2 == ge::DT_INT8) ||
        (dataType1 == ge::DT_UINT8 && dataType2 == ge::DT_UINT8)) {
        ubDataNumber = INT8_UB_NUM;
    }
    info.tileBlockNum = (ubSize / BUFFER_NUM / BLOCK_SIZE) / ubDataNumber;
    OP_CHECK_IF(info.inputBytes == 0, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    info.tileDataNum = (info.tileBlockNum * BLOCK_SIZE) / info.inputBytes;
    info.inputLengthAlign256 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    // ======== 判断标量输入 ========
    bool is_input0_scalar = (context->GetInputShape(0)->GetStorageShape().GetShapeSize() == 1);
    bool is_input1_scalar = (context->GetInputShape(1)->GetStorageShape().GetShapeSize() == 1);
    info.is_input0_scalar = is_input0_scalar;
    info.is_input1_scalar = is_input1_scalar;
    // ======== 广播后的输出形状 ========
    uint64_t x1ShapeArr[DIMS_LIMIT] = {0};
    uint64_t x2ShapeArr[DIMS_LIMIT] = {0};
    uint64_t yShapeArr[DIMS_LIMIT] = {0};
    const gert::Shape x1ShapeObj = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape x2ShapeObj = context->GetInputShape(1)->GetStorageShape();
    uint64_t dimNum1 = x1ShapeObj.GetDimNum();
    uint64_t dimNum2 = x2ShapeObj.GetDimNum();
    uint64_t dimMax = (dimNum1 > dimNum2) ? dimNum1 : dimNum2;
    uint64_t X1TotalNum = 1;
    uint64_t X2TotalNum = 1;
    for (uint64_t i = 0; i < dimNum1; ++i) {
        x1ShapeArr[i] = static_cast<uint64_t>(x1ShapeObj.GetDim(i));
        X1TotalNum *= x1ShapeArr[i];
    }
    for (uint64_t i = 0; i < dimNum2; ++i) {
        x2ShapeArr[i] = static_cast<uint64_t>(x2ShapeObj.GetDim(i));
        X2TotalNum *= x2ShapeArr[i];
    }
    info.X1TotalNum = X1TotalNum;
    info.X2TotalNum = X2TotalNum;
    // 从最后一维开始对齐（NumPy广播规则）
    for (uint32_t i = 0; i < dimMax; ++i) {
        int idx1 = dimNum1 - 1 - i;
        int idx2 = dimNum2 - 1 - i;
        uint32_t s1 = (idx1 >= 0) ? x1ShapeObj.GetDim(idx1) : 1;
        uint32_t s2 = (idx2 >= 0) ? x2ShapeObj.GetDim(idx2) : 1;
        OP_CHECK_IF(
            (s1 != s2 && s1 != 1 && s2 != 1), OP_LOGE(context, "Broadcast Fail,Please check your input shape"),
            return ge::GRAPH_FAILED);
        yShapeArr[dimMax - 1 - i] = (s1 > s2) ? s1 : s2;
    }
    // ===============================================
    // Host 端预计算广播 stride（右对齐 + 安全补齐）
    // ===============================================
    uint32_t alignedX1[DIMS_LIMIT] = {0};
    uint32_t alignedX2[DIMS_LIMIT] = {0};
    uint32_t alignedY[DIMS_LIMIT] = {0};

    for (uint32_t i = 0; i < dimMax; ++i) {
        int idx1 = i - (dimMax - dimNum1);
        alignedX1[i] = (idx1 >= 0) ? x1ShapeArr[idx1] : 1;

        int idx2 = i - (dimMax - dimNum2);
        alignedX2[i] = (idx2 >= 0) ? x2ShapeArr[idx2] : 1;

        alignedY[i] = yShapeArr[i];
    }

    // ===============================================
    // 计算输出 stride
    uint32_t strideY[DIMS_LIMIT] = {0};
    strideY[dimMax - 1] = 1;
    for (int i = dimMax - 2; i >= 0; --i) {
        strideY[i] = strideY[i + 1] * alignedY[i + 1];
    }

    // 计算输入 stride
    uint32_t strideX1[DIMS_LIMIT] = {0};
    uint32_t strideX2[DIMS_LIMIT] = {0};
    strideX1[dimMax - 1] = 1;
    strideX2[dimMax - 1] = 1;
    for (int i = dimMax - 2; i >= 0; --i) {
        strideX1[i] = strideX1[i + 1] * alignedX1[i + 1];
        strideX2[i] = strideX2[i + 1] * alignedX2[i + 1];
    }

    // 计算有效 stride（广播维置 0）
    uint32_t effStrideX1[DIMS_LIMIT] = {0};
    uint32_t effStrideX2[DIMS_LIMIT] = {0};
    for (uint32_t i = 0; i < dimMax; ++i) {
        effStrideX1[i] = (alignedX1[i] == 1) ? 0 : strideX1[i];
        effStrideX2[i] = (alignedX2[i] == 1) ? 0 : strideX2[i];
    }

    // 写入 info
    info.yDim = dimMax;
    for (uint32_t i = 0; i < DIMS_LIMIT; ++i) {
        info.strideX1[i] = effStrideX1[i];
        info.strideX2[i] = effStrideX2[i];
        info.strideY[i] = strideY[i];
    }

    // ======== 判断输入与输出形状是否一致 ========
    bool isSameX1 = true;
    bool isSameX2 = true;
    for (uint32_t i = 0; i < dimMax; ++i) {
        if (alignedX1[i] != alignedY[i])
            isSameX1 = false;
        if (alignedX2[i] != alignedY[i])
            isSameX2 = false;
    }
    info.isSameX1 = isSameX1;
    info.isSameX2 = isSameX2;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(int64_t coreNum, Pow2ShapeInfo& info, gert::TilingContext* context)
{
    OP_CHECK_IF(
        (0 == coreNum || 0 == info.tileBlockNum || 0 == info.inputBytes), OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    uint64_t everyCoreInputBlockNum = info.inputLengthAlign256 / BLOCK_SIZE / coreNum;
    info.tailBlockNum = (info.inputLengthAlign256 / BLOCK_SIZE) % coreNum;
    info.smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / info.inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalSmallTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    info.smallTailDataNum = info.smallCoreDataNum - (info.tileDataNum * smallTileNum);
    info.smallTailDataNum = info.smallTailDataNum == 0 ? info.tileDataNum : info.smallTailDataNum;

    everyCoreInputBlockNum += 1;
    info.bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / info.inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalBigTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    info.bigTailDataNum = info.bigCoreDataNum - info.tileDataNum * bigTileNum;
    info.bigTailDataNum = info.bigTailDataNum == 0 ? info.tileDataNum : info.bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Pow2TilingFunc(gert::TilingContext* context)
{
    // Pow2TilingData tiling;
    Pow2TilingData* tiling = context->GetTilingData<Pow2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(Pow2TilingData), 0, sizeof(Pow2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 获取输入数据信息
    Pow2ShapeInfo shapeInfo;
    ret = GetShapeAttrsInfo(context, ubSize, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 计算coreNum
    if (shapeInfo.tileDataNum >= shapeInfo.inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (static_cast<uint64_t>(coreNum) < shapeInfo.inputLengthAlign256 / BLOCK_SIZE) ?
                      coreNum :
                      shapeInfo.inputLengthAlign256 / BLOCK_SIZE;
    }
    // 计算每个core处理的数据块数
    ret = CalculateCoreBlockNums(coreNum, shapeInfo, context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);
    // 设置tiling数据
    tiling->smallCoreDataNum = static_cast<uint32_t>(shapeInfo.smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint32_t>(shapeInfo.bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint32_t>(shapeInfo.tileDataNum);
    tiling->smallTailDataNum = static_cast<uint32_t>(shapeInfo.smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint32_t>(shapeInfo.bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(shapeInfo.finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint32_t>(shapeInfo.finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint32_t>(shapeInfo.tailBlockNum);
    tiling->X1TotalNum = static_cast<uint32_t>(shapeInfo.X1TotalNum);
    tiling->X2TotalNum = static_cast<uint32_t>(shapeInfo.X2TotalNum);
    tiling->yDim = static_cast<uint32_t>(shapeInfo.yDim);
    tiling->is_input0_scalar = static_cast<bool>(shapeInfo.is_input0_scalar);
    tiling->is_input1_scalar = static_cast<bool>(shapeInfo.is_input1_scalar);
    tiling->isSameX1 = static_cast<bool>(shapeInfo.isSameX1);
    tiling->isSameX2 = static_cast<bool>(shapeInfo.isSameX2);
    for (uint32_t i = 0; i < DIMS_LIMIT; i++) {
        tiling->strideX1[i] = static_cast<uint32_t>(shapeInfo.strideX1[i]);
        tiling->strideX2[i] = static_cast<uint32_t>(shapeInfo.strideX2[i]);
        tiling->strideY[i] = static_cast<uint32_t>(shapeInfo.strideY[i]);
    }
    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    // 设置tilingKey.
    uint32_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Pow2).Tiling(Pow2TilingFunc).TilingParse<Pow2CompileInfo>(TilingParseForPow2);
} // namespace optiling