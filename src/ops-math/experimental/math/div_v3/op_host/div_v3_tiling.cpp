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
 * \file div_v3_tiling.cpp
 * \brief DivV3 tiling implementation
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/div_v3_tiling_data.h"
#include "../op_kernel/div_v3_tiling_key.h"

namespace optiling {

const uint32_t BUFFER_NUM = 2;

struct DivV3CompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType, int64_t& divMode)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    totalIdx = inputX->GetStorageShape().GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype for DivV3");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* modePtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, modePtr);
    divMode = *modePtr;
    if (divMode < 0 || divMode > 2) {
        OP_LOGE(context, "invalid divMode %ld, expected 0, 1, or 2", divMode);
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
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeUbTileParams(
    gert::TilingContext* context, uint64_t ubSize, uint32_t blockSize, uint64_t inputBytes, ge::DataType dataType,
    int64_t divMode, uint32_t& tileBlockNum, uint32_t& tileDataNum)
{
    uint32_t ubDataNumber = 3U; // x, y, z queues
    if (dataType != ge::DT_FLOAT) {
        ubDataNumber += 2U; // tmpBuf0, tmpBuf1
    }
    if (divMode == 2) {
        ubDataNumber += 1U; // tmpBufFloor
    }

    if (blockSize == 0 || inputBytes == 0) {
        OP_LOGE(context, "blockSize or inputBytes is 0");
        return ge::GRAPH_FAILED;
    }

    uint64_t tmp = (ubSize / blockSize) / BUFFER_NUM;
    tileBlockNum = 1U;
    if (tmp > 0) {
        uint64_t tb = tmp / ubDataNumber;
        tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
    }

    tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * blockSize) / inputBytes);
    if (tileDataNum == 0U) {
        tileDataNum = 1U;
    }
    return ge::GRAPH_SUCCESS;
}

static inline void WriteTilingData(
    DivV3TilingData* tiling, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t tileDataNum,
    uint32_t smallTailDataNum, uint32_t bigTailDataNum, uint32_t finalSmallTileNum, uint32_t finalBigTileNum,
    uint32_t tailBlockNum)
{
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);
}

static ge::graphStatus ComputeCoreDistribution(
    uint64_t inputLengthBytes, uint32_t blockSize, int64_t coreNum, uint64_t ubSize, uint64_t inputBytes,
    int64_t totalIdx, uint32_t tileBlockNum, uint32_t tileDataNum, DivV3TilingData* tiling, uint32_t& finalCoreNum)
{
    if (inputBytes == 0 || blockSize == 0 || tileBlockNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t blocksTotal = Ops::Base::CeilDiv(inputLengthBytes, static_cast<uint64_t>(blockSize));
    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (static_cast<uint64_t>(totalIdx) <= (ubSize / inputBytes / BUFFER_NUM)) {
        coreNum64 = 1;
    }
    if (coreNum64 > blocksTotal) {
        coreNum64 = blocksTotal;
    }
    if (coreNum64 == 0ULL) {
        coreNum64 = 1ULL;
    }

    finalCoreNum = static_cast<uint32_t>(coreNum64);
    uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64;
    uint32_t tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64);

    // small-core
    uint64_t smallCoreDataNum_u = everyCoreInputBlockNum * blockSize / inputBytes;
    uint32_t smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum_u);
    uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    int64_t smallTailDataNum_i =
        static_cast<int64_t>(smallCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(smallTileNum);
    uint32_t smallTailDataNum = (smallTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(smallTailDataNum_i);

    // big-core
    uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
    uint64_t bigCoreDataNum_u = bigEveryCoreBlockNum * blockSize / inputBytes;
    uint32_t bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum_u);
    uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    int64_t bigTailDataNum_i =
        static_cast<int64_t>(bigCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(bigTileNum);
    uint32_t bigTailDataNum = (bigTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(bigTailDataNum_i);

    WriteTilingData(
        tiling, smallCoreDataNum, bigCoreDataNum, tileDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum,
        finalBigTileNum, tailBlockNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DivV3TilingFunc(gert::TilingContext* context)
{
    // 1. platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);

    // 2. shapes, dtype, mode
    int64_t totalIdx = 0;
    ge::DataType dataType;
    int64_t divMode = 0;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType, divMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // handle empty input
    if (totalIdx <= 0) {
        DivV3TilingData* tiling = context->GetTilingData<DivV3TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(DivV3TilingData), 0, sizeof(DivV3TilingData));
        tiling->divMode = divMode;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    DivV3TilingData* tiling = context->GetTilingData<DivV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(DivV3TilingData), 0, sizeof(DivV3TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 4. compute type size
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return ge::GRAPH_FAILED);
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;

    // 5. UB tile parameters (extracted to sub-function)
    uint32_t tileBlockNum = 1U;
    uint32_t tileDataNum = 1U;
    OP_CHECK_IF(
        ComputeUbTileParams(context, ubSize, blockSize, inputBytes, dataType, divMode, tileBlockNum, tileDataNum) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ComputeUbTileParams error"), return ge::GRAPH_FAILED);

    // 6. multi-core distribution (extracted to sub-function)
    uint32_t finalCoreNum = 1U;
    OP_CHECK_IF(
        ComputeCoreDistribution(
            inputLengthBytes, blockSize, coreNum, ubSize, inputBytes, totalIdx, tileBlockNum, tileDataNum, tiling,
            finalCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ComputeCoreDistribution error"), return ge::GRAPH_FAILED);

    tiling->divMode = divMode;
    context->SetBlockDim(finalCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDivV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DivV3).Tiling(DivV3TilingFunc).TilingParse<DivV3CompileInfo>(TilingParseForDivV3);

} // namespace optiling
