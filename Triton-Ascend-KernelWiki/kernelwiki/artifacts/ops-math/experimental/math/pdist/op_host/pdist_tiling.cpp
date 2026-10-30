/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "experimental/math/pdist/op_kernel/pdist_tiling_data.h"
#include "experimental/math/pdist/op_kernel/pdist_tiling_key.h"
#include "experimental/math/pdist/op_kernel/pdist_constants.h"
#include <cmath>

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sizeof(float);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseInputParams(gert::TilingContext* context,
    uint32_t& rows, uint32_t& cols, ge::DataType& dataType, float& pValue)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto shape = inputShape->GetStorageShape();
    int64_t rowsI64 = shape.GetDim(0);
    int64_t colsI64 = shape.GetDim(1);
    OP_CHECK_IF(rowsI64 < 2, OP_LOGE(context, "rows must >= 2, got %ld", rowsI64), return ge::GRAPH_FAILED);
    OP_CHECK_IF(colsI64 < 1, OP_LOGE(context, "cols must >= 1, got %ld", colsI64), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rowsI64 > PDIST_MAX_SUPPORTED_ROWS,
        OP_LOGE(context, "rows %ld exceeds max supported %u", rowsI64, PDIST_MAX_SUPPORTED_ROWS),
        return ge::GRAPH_FAILED);
    rows = static_cast<uint32_t>(rowsI64);
    cols = static_cast<uint32_t>(colsI64);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    pValue = 2.0f;
    auto attrP = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrP);
    const float* pPtr = attrP->GetFloat(0);
    if (pPtr != nullptr) {
        pValue = *pPtr;
    }
    return ge::GRAPH_SUCCESS;
}

static uint32_t ComputeReduceBufSize(uint32_t count)
{
    auto shape = ge::Shape({static_cast<int64_t>(count)});
    uint32_t sumMax = 0, sumMin = 0, maxMax = 0, maxMin = 0;
    AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DT_FLOAT,
        AscendC::ReducePattern::R, true, false, sumMax, sumMin);
    AscendC::GetReduceMaxMaxMinTmpSize(shape, ge::DT_FLOAT,
        AscendC::ReducePattern::R, true, false, maxMax, maxMin);
    uint32_t result = (sumMin > maxMin) ? sumMin : maxMin;
    return ((result + 31) / 32) * 32;
}

static ge::graphStatus ComputeUbBudget(gert::TilingContext* context, uint64_t ubSize, bool isFp16,
    uint32_t& ubTensorEachLoop, uint32_t& reduceBufSize)
{
    uint32_t initialUbLoop = static_cast<uint32_t>(ubSize) / (2 * sizeof(float));
    initialUbLoop = (initialUbLoop / PDIST_DATA_EACH_BLOCK) * PDIST_DATA_EACH_BLOCK;
    reduceBufSize = ComputeReduceBufSize(initialUbLoop);

    uint32_t reservedBytes = reduceBufSize + PDIST_SUM_TENSOR_SIZE * sizeof(float)
                           + PDIST_SUM_TENSOR_SIZE * sizeof(float);
    if (isFp16) {
        reservedBytes += PDIST_SUM_TENSOR_SIZE * sizeof(uint16_t);
    }
    OP_CHECK_IF(static_cast<uint32_t>(ubSize) <= reservedBytes,
        OP_LOGE(context, "UB size %lu too small for reserved %u bytes", ubSize, reservedBytes),
        return ge::GRAPH_FAILED);
    uint32_t availableBytes = static_cast<uint32_t>(ubSize) - reservedBytes;
    uint32_t bytesPerPair = 2 * sizeof(float);
    if (isFp16) {
        bytesPerPair += sizeof(uint16_t);
    }
    ubTensorEachLoop = availableBytes / bytesPerPair;
    ubTensorEachLoop = (ubTensorEachLoop / PDIST_DATA_EACH_BLOCK) * PDIST_DATA_EACH_BLOCK;
    OP_CHECK_IF(ubTensorEachLoop == 0,
        OP_LOGE(context, "UB too small, ubTensorEachLoop is 0"), return ge::GRAPH_FAILED);

    reduceBufSize = ComputeReduceBufSize(ubTensorEachLoop);
    return ge::GRAPH_SUCCESS;
}

static void ComputeCoreSplit(uint64_t computeNum, uint32_t& usedCores,
    uint64_t& numBlockEachCore, uint64_t& lastNumsBlocks, uint64_t& lastNumsNoneFullBlock)
{
    uint64_t neededCores = (computeNum + 7) / 8;
    if (neededCores < usedCores) {
        usedCores = static_cast<uint32_t>(neededCores);
    }
    if (usedCores == 0) {
        usedCores = 1;
    }

    uint64_t dataEachBlock = 8;
    numBlockEachCore = computeNum / usedCores / dataEachBlock;
    uint64_t lastNums = computeNum % (dataEachBlock * usedCores);
    lastNumsBlocks = lastNums / dataEachBlock;
    lastNumsNoneFullBlock = lastNums % dataEachBlock;
}

static ge::graphStatus FillTilingData(gert::TilingContext* context,
    uint32_t rows, uint32_t cols, float pValue, uint64_t computeNum,
    uint32_t ubTensorEachLoop, uint32_t usedCores, uint32_t reduceBufSize,
    uint64_t numBlockEachCore, uint64_t lastNumsBlocks, uint64_t lastNumsNoneFullBlock,
    ge::DataType dataType)
{
    uint64_t tilingKey = 1;
    if (pValue == 0.0f) {
        tilingKey = 0;
    } else if (std::isinf(pValue)) {
        tilingKey = 2;
    }

    PdistTilingData* tiling = context->GetTilingData<PdistTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(PdistTilingData), 0, sizeof(PdistTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->rows = rows;
    tiling->cols = cols;
    tiling->pValue = pValue;
    tiling->computeNum = computeNum;
    tiling->ubTensorEachLoop = ubTensorEachLoop;
    tiling->coreNumVar = usedCores;
    tiling->tilingKey = static_cast<uint32_t>(tilingKey);
    tiling->reduceBufSize = reduceBufSize;
    tiling->numBlockEachCore = numBlockEachCore;
    tiling->lastNumsBlocks = lastNumsBlocks;
    tiling->lastNumsNoneFullBlock = lastNumsNoneFullBlock;

    context->SetBlockDim(usedCores);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PdistTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    uint32_t rows, cols;
    ge::DataType dataType;
    float pValue;
    OP_CHECK_IF(
        ParseInputParams(context, rows, cols, dataType, pValue) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ParseInputParams error"), return ge::GRAPH_FAILED);

    bool isFp16 = (dataType == ge::DT_FLOAT16);
    uint32_t ubTensorEachLoop, reduceBufSize;
    OP_CHECK_IF(
        ComputeUbBudget(context, ubSize, isFp16, ubTensorEachLoop, reduceBufSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ComputeUbBudget error"), return ge::GRAPH_FAILED);

    uint64_t computeNum = static_cast<uint64_t>(rows) * (rows - 1) / 2;
    uint32_t usedCores = static_cast<uint32_t>(coreNum);
    uint64_t numBlockEachCore, lastNumsBlocks, lastNumsNoneFullBlock;
    ComputeCoreSplit(computeNum, usedCores, numBlockEachCore, lastNumsBlocks, lastNumsNoneFullBlock);

    return FillTilingData(context, rows, cols, pValue, computeNum,
        ubTensorEachLoop, usedCores, reduceBufSize,
        numBlockEachCore, lastNumsBlocks, lastNumsNoneFullBlock, dataType);
}

static ge::graphStatus TilingParseForPdist([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PdistCompileInfo {};

IMPL_OP_OPTILING(Pdist)
    .Tiling(PdistTilingFunc)
    .TilingParse<PdistCompileInfo>(TilingParseForPdist);

} // namespace optiling
