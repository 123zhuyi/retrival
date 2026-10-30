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
 * \file data_compare_tiling.cpp
 * \brief DataCompare Host 侧 Tiling 实现（DAV_3510 / arch35）
 *
 * All Reduce 算子：
 *   - 合轴后 pattern = AR（axisNum=2, axisShape=[1, totalElements]）
 *   - isTailR 恒为 true
 *   - 3 套 TilingKey：Normal / EMPTY / Group
 *   - Reducer = ReduceSum（identity=0, needs_bisection=true, is_fast_path=true）
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/data_compare_tiling_data.h"
#include "../../op_kernel/arch35/data_compare_tiling_key.h"
#include "data_compare_tiling.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t CACHE_BUF_SIZE = 16 * 1024; // 16 KB

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum, uint64_t* blockSize, uint64_t* cacheLineSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    *blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(*blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);

    *cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(*cacheLineSize == 0, OP_LOGE(context, "cacheLineSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

struct ShapeAttrsInfo {
    int64_t totalElements = 1;
    int64_t typeSize = 4;
    float atol = 1e-5f;
    float rtol = 1e-3f;
    bool isEmptyTensor = false;
};

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, ShapeAttrsInfo* info)
{
    auto inputX1 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX1);
    auto inputX2 = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX2);

    auto x1Shape = inputX1->GetStorageShape();
    auto x2Shape = inputX2->GetStorageShape();

    int64_t x1Rank = static_cast<int64_t>(x1Shape.GetDimNum());
    int64_t x2Rank = static_cast<int64_t>(x2Shape.GetDimNum());
    OP_CHECK_IF(
        x1Rank != x2Rank, OP_LOGE(context, "x1 rank %ld != x2 rank %ld", x1Rank, x2Rank), return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < x1Rank; ++i) {
        int64_t d1 = x1Shape.GetDim(static_cast<size_t>(i));
        int64_t d2 = x2Shape.GetDim(static_cast<size_t>(i));
        OP_CHECK_IF(
            d1 != d2, OP_LOGE(context, "x1 dim[%ld]=%ld != x2 dim[%ld]=%ld", i, d1, i, d2), return ge::GRAPH_FAILED);
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* atolPtr = attrs->GetAttrPointer<float>(0);
    const float* rtolPtr = attrs->GetAttrPointer<float>(1);
    info->atol = (atolPtr != nullptr) ? *atolPtr : 1e-5f;
    info->rtol = (rtolPtr != nullptr) ? *rtolPtr : 1e-3f;

    for (int64_t i = 0; i < x1Rank; ++i) {
        int64_t dim = x1Shape.GetDim(static_cast<size_t>(i));
        if (dim == 0) {
            info->isEmptyTensor = true;
            break;
        }
        info->totalElements *= dim;
    }
    if (x1Rank == 0) {
        info->totalElements = 1;
    }

    switch (dataType) {
        case ge::DT_FLOAT:
            info->typeSize = 4;
            break;
        case ge::DT_FLOAT16:
            info->typeSize = 2;
            break;
        case ge::DT_BF16:
            info->typeSize = 2;
            break;
        case ge::DT_INT8:
            info->typeSize = 1;
            break;
        case ge::DT_UINT8:
            info->typeSize = 1;
            break;
        case ge::DT_INT32:
            info->typeSize = 4;
            break;
        default:
            OP_LOGE(context, "Unsupported dtype: %d", static_cast<int>(dataType));
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleEmptyTensor(
    gert::TilingContext* context, DataCompareTilingData* tiling, const ShapeAttrsInfo* info)
{
    tiling->usedCoreNum = 0;
    tiling->axisShape[0] = info->totalElements;
    tiling->atol = info->atol;
    tiling->rtol = info->rtol;
    tiling->cacheBufUbSize = CACHE_BUF_SIZE;

    context->SetBlockDim(1);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(0), static_cast<uint32_t>(1));

    size_t sysWsSize = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetLibApiWorkSpaceSize();
    size_t* wsSizes = context->GetWorkspaceSizes(WORKSPACE_NUM);
    wsSizes[0] = sysWsSize;

    OP_LOGI(context->GetNodeName(), "EMPTY tensor path, totalElements=0");
    return ge::GRAPH_SUCCESS;
}

// All Reduce 算子合轴（简化版）
// 对于 All Reduce，pattern 固定为 AR，只需设置 axisNum/axisShape/axisStride
static void FuseAxis(DataCompareTilingData* tiling, int64_t totalElements)
{
    tiling->axisNum = 2;
    tiling->axisShape[0] = 1;             // A轴
    tiling->axisShape[1] = totalElements; // R轴
    tiling->axisStride[0] = totalElements;
    tiling->axisStride[1] = 1;
}

static ge::graphStatus ComputeUbFactorTailR(
    gert::TilingContext* context, DataCompareTilingData* tiling, uint64_t ubSize, uint64_t blockSize, int64_t typeSize,
    int64_t totalElements)
{
    // All Reduce 恒为 tail-R
    tiling->aUbFactor = 1;
    tiling->aUbFactorAlign = 1;
    tiling->innerAProd = 1;
    tiling->innerAProdAlign = 1;
    tiling->innerRProd = 1;
    tiling->innerRProdAlign = 1;

    int64_t ubAvailable = static_cast<int64_t>(ubSize) - CACHE_BUF_SIZE;
    int64_t bsElem = static_cast<int64_t>(blockSize) / typeSize;

    int64_t aUnit = tiling->aUbFactorAlign * tiling->innerAProdAlign;
    int64_t aOnlyBytes = aUnit * typeSize;
    int64_t bytesPerRElem = 3 * typeSize + 2 * static_cast<int64_t>(sizeof(float));
    int64_t r_i_max = (ubAvailable - aOnlyBytes) / (aUnit * bytesPerRElem);
    if (r_i_max < 1) {
        OP_LOGE(context, "UB too small: r_i_max=%ld", r_i_max);
        return ge::GRAPH_FAILED;
    }

    int64_t rUbFactor = r_i_max;
    if (rUbFactor > totalElements) {
        rUbFactor = totalElements;
    }

    if (rUbFactor < totalElements) {
        rUbFactor = FloorAlign(rUbFactor, bsElem);
        if (rUbFactor == 0) {
            OP_LOGE(context, "rUbFactor FloorAlign to 0, bsElem=%ld", bsElem);
            return ge::GRAPH_FAILED;
        }
    }

    int64_t rUbFactorAlign = CeilAlign(rUbFactor, bsElem);

    tiling->rUbFactor = rUbFactor;
    tiling->rUbFactorAlign = rUbFactorAlign;

    return ge::GRAPH_SUCCESS;
}

static void ComputeUbSizes(DataCompareTilingData* tiling, int64_t typeSize, uint64_t blockSize)
{
    int64_t unit = tiling->rUbFactorAlign;
    tiling->preReduceUbSize = CeilAlign(unit * typeSize, static_cast<int64_t>(blockSize));
    tiling->postReduceUbSize = CeilAlign(static_cast<int64_t>(sizeof(float)), static_cast<int64_t>(blockSize));
    tiling->tmpBufUbSize = CeilAlign(unit * static_cast<int64_t>(sizeof(float)), static_cast<int64_t>(blockSize));
    tiling->cacheBufUbSize = CACHE_BUF_SIZE;
}

static void ComputeALoopCntTotal(DataCompareTilingData* tiling)
{
    int64_t aSplitChunkCnt = CeilDiv(tiling->axisShape[0], tiling->aUbFactor);
    int64_t aLoopCntTotal = aSplitChunkCnt;
    tiling->aSplitChunkCnt = aSplitChunkCnt;
    tiling->aLoopCntTotal = aLoopCntTotal;
}

static void ComputeCoreBalance(DataCompareTilingData* tiling, int64_t coreNum)
{
    int64_t aLoopCntTotal = tiling->aLoopCntTotal;
    int64_t aSmallCoreLoopCnt = aLoopCntTotal / coreNum;
    int64_t aBigCoreCnt = aLoopCntTotal % coreNum;
    int64_t aBigCoreLoopCnt = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
    int32_t usedCoreNum = (aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(coreNum) : static_cast<int32_t>(aBigCoreCnt);

    tiling->aSmallCoreLoopCnt = aSmallCoreLoopCnt;
    tiling->aBigCoreCnt = static_cast<int32_t>(aBigCoreCnt);
    tiling->aBigCoreLoopCnt = aBigCoreLoopCnt;
    tiling->usedCoreNum = usedCoreNum;
}

static bool ShouldUseGroup(int64_t aLoopCntTotal, int64_t rLoopCntTotal, int64_t coreNum)
{
    return (aLoopCntTotal <= coreNum / 2 && rLoopCntTotal > 1);
}

static ge::graphStatus ComputeGroupSplit(
    gert::TilingContext* context, DataCompareTilingData* tiling, int64_t coreNum, int64_t rLoopCntTotal,
    int32_t* usedCoreNum)
{
    int64_t aOuter = tiling->aLoopCntTotal;
    int64_t rOuter = rLoopCntTotal;

    int64_t totalOuter = aOuter * rOuter;
    int64_t perCoreNum = CeilDiv(totalOuter, coreNum);
    int64_t numBlocks = CeilDiv(totalOuter, perCoreNum);

    int64_t ceilAligned = CeilAlign(numBlocks, aOuter);
    if (ceilAligned <= coreNum) {
        numBlocks = ceilAligned;
    } else {
        numBlocks = FloorAlign(numBlocks, aOuter);
    }
    if (numBlocks < 1)
        numBlocks = 1;

    *usedCoreNum = static_cast<int32_t>(numBlocks);
    tiling->usedCoreNum = *usedCoreNum;

    int64_t rGroupCnt = numBlocks / aOuter;
    tiling->rGroupCnt = rGroupCnt;

    OP_CHECK_IF(
        context->SetScheduleMode(1) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Failed to set ScheduleMode"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void SetWorkspaceSize(gert::TilingContext* context, const DataCompareTilingData* tiling, bool isGroup)
{
    size_t sysWsSize = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetLibApiWorkSpaceSize();
    size_t usrWsSize = 0;
    if (isGroup) {
        int64_t aTotal = 1;
        for (int32_t i = 0; i < tiling->axisNum; i += 2) {
            aTotal *= tiling->axisShape[i];
        }
        usrWsSize = static_cast<size_t>(tiling->rGroupCnt) * static_cast<size_t>(aTotal) * sizeof(float);
        usrWsSize = ((usrWsSize + 31) / 32) * 32;
    }
    size_t* wsSizes = context->GetWorkspaceSizes(WORKSPACE_NUM);
    wsSizes[0] = usrWsSize + sysWsSize;
}

static ge::graphStatus DataCompareTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin DataCompareTilingFunc");

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    uint64_t blockSize = 0;
    uint64_t cacheLineSize = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum, &blockSize, &cacheLineSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    ShapeAttrsInfo info;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &info) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo failed"),
        return ge::GRAPH_FAILED);

    DataCompareTilingData* tiling = context->GetTilingData<DataCompareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(DataCompareTilingData), 0, sizeof(DataCompareTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (info.isEmptyTensor) {
        return HandleEmptyTensor(context, tiling, &info);
    }

    // pattern 预处理（All Reduce 只需 FuseAxis）
    FuseAxis(tiling, info.totalElements);

    tiling->aSplitAxisIdx = 0;
    tiling->rSplitAxisIdx = 1;

    OP_CHECK_IF(
        ComputeUbFactorTailR(context, tiling, ubSize, blockSize, info.typeSize, info.totalElements) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ComputeUbFactorTailR failed"), return ge::GRAPH_FAILED);

    ComputeUbSizes(tiling, info.typeSize, blockSize);

    int64_t rLoopCntTotal = CeilDiv(info.totalElements, tiling->rUbFactor);
    tiling->rLoopCntTotal = rLoopCntTotal;

    ComputeALoopCntTotal(tiling);
    ComputeCoreBalance(tiling, coreNum);

    int32_t templateType = 0;
    int32_t usedCoreNum = tiling->usedCoreNum;
    if (ShouldUseGroup(tiling->aLoopCntTotal, rLoopCntTotal, coreNum)) {
        templateType = 1;
        OP_CHECK_IF(
            ComputeGroupSplit(context, tiling, coreNum, rLoopCntTotal, &usedCoreNum) != ge::GRAPH_SUCCESS,
            OP_LOGE(context, "ComputeGroupSplit failed"), return ge::GRAPH_FAILED);
    }

    tiling->atol = info.atol;
    tiling->rtol = info.rtol;

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(templateType), static_cast<uint32_t>(0));
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    bool isGroup = (templateType == 1);
    SetWorkspaceSize(context, tiling, isGroup);

    OP_LOGI(
        context->GetNodeName(),
        "Tiling done: totalElements=%ld, rUbFactor=%ld, rLoopCntTotal=%ld, "
        "usedCoreNum=%d, isGroup=%d, templateType=%d",
        info.totalElements, tiling->rUbFactor, rLoopCntTotal, usedCoreNum, static_cast<int>(isGroup), templateType);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DataCompare(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DataCompareCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DataCompare)
    .Tiling(DataCompareTilingFunc)
    .TilingParse<DataCompareCompileInfo>(TilingPrepare4DataCompare);

} // namespace optiling
