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
 * \file isin_part_v1_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/isin_part_v1_tiling_data.h"
#include "../op_kernel/isin_part_v1_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

struct IsinPartV1CompileInfo {};

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
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
    auto inputShapeX = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());
    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = Ops::Base::EnsureNotScalar(inputY->GetStorageShape());
    auto inputElementsNum = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputElementsNum);
    auto inputShapeElementsNum = Ops::Base::EnsureNotScalar(inputElementsNum->GetStorageShape());
    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
    auto outShapeZ = Ops::Base::EnsureNotScalar(outZ->GetStorageShape());

    // shape校验
    bool shapeMatch = true;

    if (inputShapeElementsNum.GetDimNum() != 1) {
        shapeMatch = false;
        // 形状不匹配则报错
        OP_CHECK_IF(
            !shapeMatch,
            OP_LOGE(
                context, "IsinPartV1: elementsNum is invalid, input dim is %zu, support dim is 1",
                inputShapeElementsNum.GetDimNum()),
            return ge::GRAPH_FAILED);
    }

    // 校验输入和输出的维度数一致
    if (inputShapeX.GetDimNum() != inputShapeY.GetDimNum() || inputShapeX.GetDimNum() != outShapeZ.GetDimNum()) {
        shapeMatch = false;
        // 形状不匹配则报错
        OP_CHECK_IF(
            !shapeMatch,
            OP_LOGE(
                context, "IsinPartV1: inputx, inputy, outputz dim num not match! dim num: x=%zu, y=%zu, z=%zu",
                inputShapeX.GetDimNum(), inputShapeY.GetDimNum(), outShapeZ.GetDimNum()),
            return ge::GRAPH_FAILED);
    } else {
        // 校验每个维度的大小一致
        size_t dimNum = inputShapeX.GetDimNum();
        for (size_t i = 0; i < dimNum; i++) {
            if (inputShapeX.GetDim(i) != inputShapeY.GetDim(i) || inputShapeX.GetDim(i) < outShapeZ.GetDim(i)) {
                shapeMatch = false;
                OP_CHECK_IF(
                    !shapeMatch,
                    OP_LOGE(
                        context, "IsinPartV1: inputx, inputy, outputz dim(%zu) not match! dim num: x=%zu, y=%zu, z=%zu",
                        i, inputShapeX.GetDimNum(), inputShapeY.GetDimNum(), outShapeZ.GetDimNum()),
                    return ge::GRAPH_FAILED);
                break;
            }
        }
    }

    // 计算总元素数量
    totalIdx = inputX->GetOriginShape().GetShapeSize();
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32};
    auto inputDescX = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescX);
    auto dataTypeX = inputDescX->GetDataType();
    if (supportedDtype.count(dataTypeX) == 0) {
        OP_LOGE(context, "the first input dtype is invalid , support dype is [float32, int32]");
        return ge::GRAPH_FAILED;
    }

    auto inputDescY = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescY);
    auto dataTypeY = inputDescY->GetDataType();
    if (supportedDtype.count(dataTypeY) == 0) {
        OP_LOGE(context, "the second input dtype is invalid , support dype is [int32]");
        return ge::GRAPH_FAILED;
    }

    auto inputDescElementsNum = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescElementsNum);
    auto dataTypeElementsNum = inputDescElementsNum->GetDataType();
    if (supportedDtype.count(dataTypeElementsNum) == 0) {
        OP_LOGE(context, "the third input dtype is invalid , support dype is [int32]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
// 可直接替换你的 IsinPartV1TilingFunc 内部实现（保留函数签名）
static ge::graphStatus IsinPartV1TilingFunc(gert::TilingContext* context)
{
    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. shapes & dtype
    int64_t totalIdx = 0;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // handle empty input
    if (totalIdx <= 0) {
        IsinPartV1TilingData* tiling = context->GetTilingData<IsinPartV1TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(IsinPartV1TilingData), 0, sizeof(IsinPartV1TilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    IsinPartV1TilingData* tiling = context->GetTilingData<IsinPartV1TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(IsinPartV1TilingData), 0, sizeof(IsinPartV1TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // --- safer numeric types ---
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;

    // ub-based tileBlockNum guard (避免为0)
    uint32_t ubDataNumber = (inputBytes == 1ULL) ? 5U : 3U;
    uint64_t tmp = (ubSize / BLOCK_SIZE / BUFFER_NUM);
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
        uint64_t tb = tmp / ubDataNumber;
        tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
    }

    // 每个 tile 包含的元素数（至少 1）
    uint32_t tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * BLOCK_SIZE) / inputBytes);
    if (tileDataNum == 0U)
        tileDataNum = 1U;

    // 总 block 数（向上取整）
    uint64_t blocksTotal = (inputLengthBytes + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;
    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (coreNum64 > blocksTotal)
        coreNum64 = blocksTotal;
    if (coreNum64 == 0ULL)
        coreNum64 = 1ULL; // 最少 1 core
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);

    uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64;              // 基本块数
    uint32_t tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64); // 前 tailBlockNum 个核是 big-core

    // small-core 数量（元素）
    uint64_t smallCoreDataNum_u = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum_u);

    uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    int64_t smallTailDataNum_i =
        static_cast<int64_t>(smallCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(smallTileNum);
    uint32_t smallTailDataNum = (smallTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(smallTailDataNum_i);

    // big-core（每个多一个 block）
    uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
    uint64_t bigCoreDataNum_u = bigEveryCoreBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum_u);
    uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    int64_t bigTailDataNum_i =
        static_cast<int64_t>(bigCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(bigTileNum);
    uint32_t bigTailDataNum = (bigTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(bigTailDataNum_i);

    // write back
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);

    tiling->totalLength = totalIdx;
    context->SetBlockDim(finalCoreNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForIsinPartV1([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(IsinPartV1).Tiling(IsinPartV1TilingFunc).TilingParse<IsinPartV1CompileInfo>(TilingParseForIsinPartV1);
} // namespace optiling
