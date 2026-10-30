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
 * \file pack_v2_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/pack_v2_tiling_data.h"
#include "../op_kernel/pack_v2_tiling_key.h"

namespace optiling {

const uint32_t BUFFER_NUM = 2;
const int32_t BLOCK_DIM = 8;

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

static inline uint64_t AlignUp(uint64_t x, uint64_t a)
{
    if (a == 0) {
        return x;
    }
    return (x + a - 1) / a * a;
}
struct PackV2CompileInfo {};

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
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_FLOAT16};
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
static ge::graphStatus PackV2TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    PackV2TilingData* tiling = context->GetTilingData<PackV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(PackV2TilingData), 0, sizeof(PackV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize(context);
    uint64_t bigCoreDataNum = 0;
    uint64_t bigCoreLoopNum = 0;
    uint64_t bigCoreTailDataNum = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = BLOCK_DIM;
    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
    uint64_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), dataTypeLength);
    uint64_t inputLength = inputDataNum * dataTypeLength;
    if (coreNum == 0 || BLOCK_SIZE == 0) {
        return ge::GRAPH_FAILED;
    }

    // There are a total of 3 shared UB spaces in the input and output.
    uint64_t ubPartNum = (dataTypeLength == 1) ? 3 : 3;
    uint64_t ubPartLength = ubSize / ubPartNum / BUFFER_NUM;
    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
    uint64_t ubPartBlockNum = ubPartLength / BLOCK_SIZE;
    uint64_t ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeLength;

    // Input data for 32B alignment
    int64_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    if (ubPartDataNum >= inputDataNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (coreNum < inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;

    // Small chunks are calculated and sliced several times using the number of data on each core
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
    uint64_t smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
    smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? smallCoreLoopNum : smallCoreLoopNum + 1;
    // Tail block calculation for small chunks of data
    uint64_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
    smallCoreTailDataNum = smallCoreTailDataNum == 0 ? ubPartDataNum : smallCoreTailDataNum;

    if (0 != tailBlockNum) {
        everyCoreInputBlockNum += 1;
        bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
        bigCoreLoopNum = bigCoreDataNum / ubPartDataNum;
        bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? bigCoreLoopNum : bigCoreLoopNum + 1;
        bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);
        bigCoreTailDataNum = bigCoreTailDataNum == 0 ? ubPartDataNum : bigCoreTailDataNum;
    }
    auto attrs = context->GetAttrs();
    int32_t d = 0; // 默认值
    if (attrs) {
        const int64_t* attrA = attrs->GetInt(0);
        if (attrA != nullptr) {
            d = *attrA;
        }
    }
    const auto xShape = context->GetInputTensor(0)->GetOriginShape();
    const auto yShape = context->GetInputTensor(1)->GetOriginShape();
    context->SetBlockDim(BLOCK_DIM);
    uint32_t totalLengthx = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    uint32_t totalLengthy = context->GetInputShape(1)->GetOriginShape().GetShapeSize();
    uint32_t totalLengthz = totalLengthy + totalLengthx;
    int32_t dimNum = xShape.GetDimNum();
    std::vector<int> dimarrX(dimNum, 0);
    std::vector<int> dimarrY(dimNum, 0);
    std::vector<int> dimarr(dimNum, 0);
    uint32_t z2 = 0;
    uint32_t x2 = xShape.GetDim(dimNum - 1); // x的列数
    uint32_t x1 = 1;
    uint32_t y2 = yShape.GetDim(dimNum - 1); // y的列数
    uint32_t y1 = 1;
    for (int i = 0; i < dimNum; i++) {
        dimarrX[i] = xShape.GetDim(i);
        dimarrY[i] = yShape.GetDim(i);
    }
    // 总的行数
    for (int i = 0; i < dimNum - 1; i++) {
        x1 = x1 * dimarrX[i];
        y1 = y1 * dimarrY[i];
    }
    for (int i = 0; i < dimNum - 1; i++) {
        if (i == d) {
            dimarr[i] = dimarrX[i] + dimarrY[i];
        } else {
            dimarr[i] = dimarrX[i];
        }
    }
    // 核间划分//1
    uint32_t big_core_num = x1 % BLOCK_DIM;
    uint32_t small_core_num = BLOCK_DIM - big_core_num;
    uint32_t small_tile_length = x1 / BLOCK_DIM;   // 小核分到的行数
    uint32_t big_tile_length = x1 / BLOCK_DIM + 1; // 大核分到的行数
    // 0
    uint32_t sbig_core_num = (x1 + y1) % BLOCK_DIM;
    uint32_t ssmall_core_num = BLOCK_DIM - sbig_core_num;
    uint32_t ssmall_tile_length = (x1 + y1) / BLOCK_DIM;   // 小核分到的行数
    uint32_t sbig_tile_length = (x1 + y1) / BLOCK_DIM + 1; // 大核分到的行数

    // 核内划分//1维度的张量
    int64_t core_tile_x1 = 1; // 对行维度进行划分
    auto FitsUB = [&](int64_t x, int64_t y, int64_t tile_x1) -> bool {
        uint64_t xBytes = AlignUp((uint64_t)tile_x1 * x * dataTypeLength, BLOCK_SIZE);
        uint64_t yBytes = AlignUp((uint64_t)y * tile_x1 * dataTypeLength, BLOCK_SIZE);
        // 总 UB = 双缓冲 * (x + z + y + temp)
        uint64_t total = BUFFER_NUM * (xBytes + yBytes) * 2;
        // 预留 5% 余量
        return total <= (ubSize * 95 / 100);
    };

    // 调整核内分块参数以适应UB
    while (FitsUB(x2, y2, core_tile_x1) && core_tile_x1 < big_tile_length) {
        core_tile_x1 *= 2;
    }
    if (core_tile_x1 != 1) { // 判断是否倍增了 回溯到倍增前的结果
        core_tile_x1 /= 2;
    }
    if (!FitsUB(x2, y2, core_tile_x1)) {
        return ge::GRAPH_FAILED;
    } //
    int64_t core_tile_s1 = 1; // 对行维度进行划分

    // 调整核内分块参数以适应UB
    while (FitsUB(x2, y2, core_tile_s1) && core_tile_s1 < sbig_tile_length) {
        core_tile_s1 *= 2;
    }
    if (core_tile_s1 != 1) { // 判断是否倍增了 回溯到倍增前的结果
        core_tile_s1 /= 2;
    }
    if (!FitsUB(x2, y2, core_tile_s1)) {
        return ge::GRAPH_FAILED;
    } //

    uint32_t small_tile_times = small_tile_length / core_tile_x1;
    uint32_t big_tile_times = big_tile_length / core_tile_x1;

    uint32_t small_tail_num = small_tile_length % core_tile_x1;
    uint32_t big_tail_num = big_tile_length % core_tile_x1;
    // 0
    uint32_t ssmall_tile_times = ssmall_tile_length / core_tile_s1;
    uint32_t sbig_tile_times = sbig_tile_length / core_tile_s1;

    uint32_t ssmall_tail_num = ssmall_tile_length % core_tile_s1;
    uint32_t sbig_tail_num = sbig_tile_length % core_tile_s1;

    // 1
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
    // 0
    if (ssmall_tail_num != 0) {
        ssmall_tile_times++;
    } else {
        ssmall_tail_num = core_tile_s1;
    }
    if (sbig_tail_num != 0) {
        sbig_tile_times++;
    } else {
        sbig_tail_num = core_tile_s1;
    }

    if (d == dimNum - 1) {
        z2 = x2 + y2;
    } else {
        z2 = x2;
    }
    uint32_t partnumX = 1;
    uint32_t partnum = 1;
    if (d == dimNum - 1) {
        partnum = 2;
        partnumX = 1;
    } else {
        for (int i = d; i < dimNum - 1; i++) {
            partnum = partnum * dimarr[i];
            partnumX = partnumX * dimarrX[i];
        }
    }
    // 计算每个核心的x和y的起始和结束以及处理的长度
    int startX[BLOCK_DIM] = {};
    int endX[BLOCK_DIM] = {};
    int rowsX[BLOCK_DIM] = {};
    int startY[BLOCK_DIM] = {};
    int endY[BLOCK_DIM] = {};
    int rowsY[BLOCK_DIM] = {};

    startX[0] = 0;
    endX[BLOCK_DIM - 1] = x1 - 1; // 最后的行数
    startY[0] = 0;
    endY[BLOCK_DIM - 1] = y1 - 1; // 最后的行数
    for (uint32_t i = 0; i < BLOCK_DIM; i++) {
        if (i < sbig_core_num) {
            int x_need_rows = 0;
            int y_need_rows = 0;
            uint32_t start = sbig_tile_length * i;
            uint32_t end = start + sbig_tile_length;
            for (uint32_t j = start; j < end; j++) {
                uint32_t temp = j % partnum;
                if (temp < partnumX) {
                    x_need_rows++;
                } else {
                    y_need_rows++;
                }
            }
            rowsX[i] = x_need_rows;
            rowsY[i] = y_need_rows;
        } else {
            int x_need_rows = 0;
            int y_need_rows = 0;
            uint32_t start = ssmall_tile_length * i + sbig_core_num;
            uint32_t end = start + ssmall_tile_length;
            for (uint32_t k = start; k < end; k++) {
                uint32_t temp = k % partnum;
                if (temp < partnumX) {
                    x_need_rows++;
                } else {
                    y_need_rows++;
                }
            }
            rowsX[i] = x_need_rows;
            rowsY[i] = y_need_rows;
        }
    }
    // 计算出开始和结束的地点
    for (int i = 0; i < BLOCK_DIM - 1; i++) {
        endX[i] = startX[i] + rowsX[i];
        startX[i + 1] = endX[i];
        endY[i] = startY[i] + rowsY[i];
        startY[i + 1] = endY[i];
    }
    // 0维度划分
    tiling->small_tile_times = small_tile_times;
    tiling->big_tile_times = big_tile_times;
    tiling->small_tail_num = small_tail_num;
    tiling->big_tail_num = big_tail_num;

    tiling->totalLengthx = totalLengthx;
    tiling->totalLengthy = totalLengthy;
    tiling->totalLengthz = totalLengthz;
    tiling->x1 = x1;
    tiling->x2 = x2;
    tiling->y1 = y1;
    tiling->y2 = y2;
    tiling->z2 = z2;
    tiling->big_core_num = static_cast<uint32_t>(big_core_num);
    tiling->small_core_num = static_cast<uint32_t>(small_core_num);

    tiling->small_tile_length = static_cast<uint32_t>(small_tile_length);
    tiling->big_tile_length = static_cast<uint32_t>(big_tile_length);
    tiling->core_tile_x1 = static_cast<uint32_t>(core_tile_x1);
    tiling->ssmall_tile_times = static_cast<uint32_t>(ssmall_tile_times);
    tiling->sbig_tile_times = static_cast<uint32_t>(sbig_tile_times);
    tiling->ssmall_tail_num = static_cast<uint32_t>(ssmall_tail_num);
    tiling->sbig_tail_num = static_cast<uint32_t>(sbig_tail_num);
    tiling->sbig_core_num = static_cast<uint32_t>(sbig_core_num);
    tiling->ssmall_core_num = static_cast<uint32_t>(ssmall_core_num);

    tiling->ssmall_tile_length = static_cast<uint32_t>(ssmall_tile_length);
    tiling->sbig_tile_length = static_cast<uint32_t>(sbig_tile_length);

    tiling->core_tile_s1 = static_cast<uint32_t>(core_tile_s1);

    tiling->partnum = static_cast<uint32_t>(partnum);
    tiling->partnumX = static_cast<uint32_t>(partnumX);
    for (int i = 0; i < BLOCK_DIM; ++i) {
        tiling->startX[i] = static_cast<int32_t>(startX[i]);
        tiling->endX[i] = static_cast<int32_t>(endX[i]);
        tiling->rowsX[i] = static_cast<int32_t>(rowsX[i]);

        tiling->startY[i] = static_cast<int32_t>(startY[i]);
        tiling->endY[i] = static_cast<int32_t>(endY[i]);
        tiling->rowsY[i] = static_cast<int32_t>(rowsY[i]);
    }
    tiling->inputDataNum = static_cast<uint64_t>(inputDataNum);
    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->ubPartDataNum = static_cast<uint64_t>(ubPartDataNum);
    tiling->smallCoreTailDataNum = static_cast<uint64_t>(smallCoreTailDataNum);
    tiling->bigCoreTailDataNum = static_cast<uint64_t>(bigCoreTailDataNum);
    tiling->smallCoreLoopNum = static_cast<uint64_t>(smallCoreLoopNum);
    tiling->bigCoreLoopNum = static_cast<uint64_t>(bigCoreLoopNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);
    tiling->d = static_cast<int32_t>(d);
    tiling->dimNum = static_cast<int32_t>(dimNum);

    uint32_t tilingKey = 0;
    if (d == dimNum) {
        tilingKey = GET_TPL_TILING_KEY(PACK_LAST);
    } else {
        tilingKey = GET_TPL_TILING_KEY(PACK_NORMAL);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(PackV2).Tiling(PackV2TilingFunc);
} // namespace optiling