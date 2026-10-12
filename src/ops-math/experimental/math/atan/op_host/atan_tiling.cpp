/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file atan_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/atan_tiling_data.h"
#include "../op_kernel/atan_tiling_key.h"

namespace optiling {

constexpr uint32_t ALIGN_SIZE = 256; // 256数据对齐
constexpr uint32_t UB_DATA_NUMBER_BF16 = 19;
constexpr uint32_t UB_DATA_NUMBER_FLOAT = 10;
constexpr uint32_t UB_DATA_NUMBER_FLOAT16 = 19;
const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

struct AtanCompileInfo {};

// 获取平台信息如coreNum, wsSysSize
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum, uint32_t& wsSysSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    wsSysSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(wsSysSize == 0, OP_LOGE(context, "wsSysSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取ubsize
static ge::graphStatus GetUBSize(gert::TilingContext* context, uint64_t& ubSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, ge::DataType& dataType, int64_t& inputLength, int32_t& inputBytes,
    int32_t& dataNumber)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
    auto inputShapeX = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());
    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto inputNum = inputShapeX.GetShapeSize(); // 输入数量

    // dtype校验
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    dataNumber = 1; // UB需要切分成几块
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }

    // dtype长度和输入长度
    inputBytes = GetSizeByDataType(dataType); // 输入类型
    inputLength = inputBytes * inputNum;      // 输入长度

    // 数据块数
    if (dataType == ge::DT_BF16) {
        dataNumber = UB_DATA_NUMBER_BF16;
    } else if (dataType == ge::DT_FLOAT) {
        dataNumber = UB_DATA_NUMBER_FLOAT;
    } else if (dataType == ge::DT_FLOAT16) {
        dataNumber = UB_DATA_NUMBER_FLOAT16;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, uint32_t& wsSysSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = wsSysSize;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus AtanTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    uint32_t wsSysSize;
    OP_CHECK_IF(
        GetPlatformInfo(context, coreNum, wsSysSize) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetUBSize(context, ubSize) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetUBSize error"), return ge::GRAPH_FAILED);

    // 2、获取shape、属性信息
    ge::DataType dataType;
    int64_t inputLength;
    int32_t inputBytes;
    int32_t dataNumber;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, dataType, inputLength, inputBytes, dataNumber) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context, wsSysSize) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    AtanTilingData* tiling = context->GetTilingData<AtanTilingData>();
    uint32_t tileBlockNum =
        (ubSize / ALIGN_SIZE) / dataNumber; // 每个ub段可用的空间块数，这时候还没有对齐，空间可以有一些冗余
    uint32_t tileDataNum = tileBlockNum * (ALIGN_SIZE / inputBytes); // 每次处理的数据量,保证数据都可以在UB里

    // 对齐后总量
    uint32_t inputLengthAlgin32 = (((inputLength + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE); // 输入长度 对齐处理
    coreNum = (coreNum < inputLengthAlgin32 / ALIGN_SIZE) ? coreNum : inputLengthAlgin32 / ALIGN_SIZE;
    coreNum = (coreNum >= 1) ? coreNum : 1;
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / ALIGN_SIZE / coreNum; // 输入数据需要多少空间块
    uint32_t tailBlockNum = inputLengthAlgin32 / ALIGN_SIZE % coreNum;           // 对齐空间后的输入数量

    // 求循环次数和尾数
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * ALIGN_SIZE / inputBytes; // 对齐空间后的输入数量
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1; // 需要循环处理几次
    uint32_t smallTailDataNum =
        smallCoreDataNum -
        (tileDataNum * smallTileNum); // 如果TileNum次处理的数据和对齐后处理的数据一样，也就是没有尾巴
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum; // 最后一次需要处理的数据量

    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * ALIGN_SIZE / inputBytes; // 对齐空间后的输入数量
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1; // 需要循环处理几次
    uint32_t bigTailDataNum =
        bigCoreDataNum - (tileDataNum * bigTileNum); // 如果TileNum次处理的数据和对齐后处理的数据一样，也就是没有尾巴
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum; // 最后一次需要处理的数据量

    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AtanTilingData), 0, sizeof(AtanTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->smallCoreDataNum = smallCoreDataNum;   // 小核输入字段，小核数据的最小大小
    tiling->bigCoreDataNum = bigCoreDataNum;       // 大核输入字段，大核数据的最小大小
    tiling->finalBigTileNum = finalBigTileNum;     // 添加输入字段，最终大核的tile数量
    tiling->finalSmallTileNum = finalSmallTileNum; // 添加输入字段，最终小核的tile数量
    tiling->smallTailDataNum = smallTailDataNum;   // 添加输入字段，小核的尾数据大小
    tiling->bigTailDataNum = bigTailDataNum;       // 添加输入字段，大核的尾数据大小
    tiling->tileDataNum = tileDataNum;             // 添加输入字段，tile数据大小
    tiling->tailBlockNum = tailBlockNum;           // 添加输入字段，大核的尾块数

    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAtan([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Atan).Tiling(AtanTilingFunc).TilingParse<AtanCompileInfo>(TilingParseForAtan);
} // namespace optiling
