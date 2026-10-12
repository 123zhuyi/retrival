/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Zhi <@hitLeechi>
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
 * \file range_infer.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/range_tiling_data.h"
#include "../op_kernel/range_tiling_key.h"

namespace optiling {
const size_t startIdx = 0;
const size_t endIdx = 1;
const size_t stepIdx = 2;
struct RangeCompileInfo {};
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
const uint32_t UB_BLOCK_NUM = 100;
const uint32_t MAX_AVAILABLE_UB_BLOCK_NUM = UB_BLOCK_NUM / BUFFER_NUM * BUFFER_NUM;
static ge::graphStatus CheckNull(uint32_t num)
{
    if (num == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParamsCalc(
    uint32_t length, uint32_t alignNum, uint32_t& tileNum, uint32_t& tileLength, uint32_t& lastTileLength)
{
    uint32_t maxPerCoreElem = alignNum * MAX_AVAILABLE_UB_BLOCK_NUM;
    CheckNull(maxPerCoreElem);
    tileNum = length / maxPerCoreElem;
    if ((length % maxPerCoreElem == 0U) || (tileNum == 0U)) {
        if (tileNum == 0U) {
            if (alignNum == 0) {
                return ge::GRAPH_FAILED;
            }
            tileNum = (length + alignNum - 1) / alignNum;
        }
        if (length < maxPerCoreElem) {
            if (alignNum == 0)
                return ge::GRAPH_FAILED;
            uint32_t blockCount = (length + alignNum - 1) / alignNum;
            blockCount = (blockCount + BUFFER_NUM - 1) / BUFFER_NUM * BUFFER_NUM;
            tileLength = blockCount * alignNum;
            if (alignNum == 0) {
                return ge::GRAPH_FAILED;
            }
            lastTileLength = (tileNum == 1) ? length : (length % alignNum == 0 ? alignNum : length % alignNum);
        } else {
            tileLength = maxPerCoreElem;
            lastTileLength = length - (tileNum - 1) * tileLength;
        }
    } else {
        tileNum++;
        tileLength = maxPerCoreElem;
        lastTileLength = length - (tileNum - 1) * tileLength;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputVal(const gert::Tensor* tensor, ge::DataType type, float& val)
{
    switch (type) {
        case ge::DT_FLOAT:
            OP_CHECK_IF(
                tensor->GetData<float>() == nullptr, OP_LOGE("Range", "NULL Input,Please Check!"),
                return ge::GRAPH_FAILED);
            val = *tensor->GetData<float>();
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void ComputeTiling(uint32_t totalLengthAligned, uint32_t alignNum, uint32_t coreNum, RangeTilingData* tiling)
{
    CheckNull(alignNum);
    CheckNull(coreNum);
    if ((totalLengthAligned / alignNum) % coreNum == 0U) {
        uint32_t blockLength = totalLengthAligned / coreNum;
        uint32_t tileNum, tileLength, lastTileLength;
        TilingParamsCalc(blockLength, alignNum, tileNum, tileLength, lastTileLength);
        tiling->blockLength = blockLength;
        tiling->tileNum = tileNum;
        tiling->tileLength = tileLength;
        tiling->lastTileLength = lastTileLength;
        tiling->isEvenCore = 1U;
    } else {
        uint32_t totalBlockCount = totalLengthAligned / alignNum;
        uint32_t formerNum = totalBlockCount % coreNum;
        uint32_t tailNum = coreNum - formerNum;
        uint32_t formerBlockCount = (totalBlockCount + coreNum - 1) / coreNum;
        uint32_t tailBlockCount = totalBlockCount / coreNum;
        uint32_t formerLength = formerBlockCount * alignNum;
        uint32_t tailLength = tailBlockCount * alignNum;
        uint32_t fTileNum, fTileLen, fLastTileLen;
        TilingParamsCalc(formerLength, alignNum, fTileNum, fTileLen, fLastTileLen);
        uint32_t tTileNum, tTileLen, tLastTileLen;
        TilingParamsCalc(tailLength, alignNum, tTileNum, tTileLen, tLastTileLen);
        tiling->formerNum = formerNum;
        tiling->formerLength = formerLength;
        tiling->formerTileNum = fTileNum;
        tiling->formerTileLength = fTileLen;
        tiling->formerLastTileLength = fLastTileLen;
        tiling->tailNum = tailNum;
        tiling->tailLength = tailLength;
        tiling->tailTileNum = tTileNum;
        tiling->tailTileLength = tTileLen;
        tiling->tailLastTileLength = tLastTileLen;
        tiling->isEvenCore = 0U;
    }
}

static ge::graphStatus RangeTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize;
    uint32_t dataTypeSize = 4; // 输出float
    uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();

    RangeTilingData* tiling = context->GetTilingData<RangeTilingData>();
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RangeTilingData), 0, sizeof(RangeTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    auto tensorStart = context->GetInputTensor(startIdx);
    auto tensorEnd = context->GetInputTensor(endIdx);
    auto tensorStep = context->GetInputTensor(stepIdx);
    if (tensorStart == nullptr || tensorEnd == nullptr || tensorStep == nullptr) {
        return ge::GRAPH_FAILED;
    }

    float startVal = 0.0f, endVal = 0.0f, stepVal = 1.0f;
    const auto inputDataType_1 = context->GetInputDesc(startIdx)->GetDataType();
    const auto inputDataType_2 = context->GetInputDesc(endIdx)->GetDataType();
    const auto inputDataType_3 = context->GetInputDesc(stepIdx)->GetDataType();
    if (GetInputVal(tensorStart, inputDataType_1, startVal) != ge::GRAPH_SUCCESS ||
        GetInputVal(tensorEnd, inputDataType_2, endVal) != ge::GRAPH_SUCCESS ||
        GetInputVal(tensorStep, inputDataType_3, stepVal) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    float diff = std::abs(endVal - startVal);
    float stepAbs = std::abs(stepVal);
    CheckNull(stepAbs);
    CheckNull(alignNum);
    int64_t totalLength = static_cast<int64_t>(std::ceil(diff / stepAbs));
    uint32_t totalLengthAligned =
        (totalLength % alignNum == 0U) ? totalLength : ((totalLength + alignNum - 1) / alignNum) * alignNum;

    ComputeTiling(totalLengthAligned, alignNum, coreNum, tiling);

    tiling->dataTypeStart = inputDataType_1;
    tiling->dataTypeEnd = inputDataType_2;
    tiling->dataTypeStep = inputDataType_3;
    tiling->totalLength = totalLength;
    tiling->totalLengthAligned = totalLengthAligned;
    context->SetBlockDim(coreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;

    uint64_t tilingKey = GET_TPL_TILING_KEY(inputDataType_1, inputDataType_2, inputDataType_3);
    OP_CHECK_IF(
        context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetTilingKey failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRange([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(Range)
    .Tiling(RangeTilingFunc)
    .TilingInputsDataDependency({0, 1, 2})
    .TilingParse<RangeCompileInfo>(TilingParseForRange);
} // namespace optiling
