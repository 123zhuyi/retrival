/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file exp_segsum_grad_tiling.cpp
 * \brief
 */
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "exp_segsum_grad_tiling.h"

namespace optiling {

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint64_t REDUCE_SUM_SIZE = 20 * 1024;

constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t BYTE_LEN_2 = 2;
constexpr int32_t BLOCK = 32;
constexpr int32_t TAILNUM = 2;
constexpr uint32_t COMMON_TILING_KEY = 0;
constexpr uint32_t SMALL_SIZE_TILING_KEY = 1;
constexpr uint32_t TENSOR_NUM = 6;
constexpr uint8_t SCHEDULE_MODE = 1; // batchmode模式，核间同步算子需要设置该模式

class ExpSegsumGradTiling {
public:
    explicit ExpSegsumGradTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    ge::graphStatus ParseInputAttrs();
    void GetWorkSpace(uint32_t needCoreNum);
    void GetNeedCoreNum(uint32_t coreNumPlatform);
    void GetTilingKey(uint64_t ubSizePlatform);
    uint8_t GetDataTypeSize();
    void FillTilingData();

    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) -> T1;

private:
    int64_t slideSize = 0;
    ExpSegsumGradTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::Shape inputShape;

    uint16_t dataTypeSize = 4;
    int64_t batches = 1;
    int64_t tailDimLength = 1;
    int32_t batchStart[MAX_CORE_CONT] = {0};
    int32_t batchEnd[MAX_CORE_CONT] = {0};
    int32_t blockSize = 8;
    uint32_t needCoreNum = 0;
    uint32_t tilingKey = 0;
};
ge::graphStatus ExpSegsumGradTiling::ParseInputAttrs()
{
    auto srcGradOutShape = tilingContext->GetInputShape(0);
    int32_t gradOutDim = srcGradOutShape->GetStorageShape().GetDimNum();
    auto gradOutShape = srcGradOutShape->GetOriginShape();
    for (int8_t i = 0; i < gradOutDim - TAILNUM; i++) {
        batches *= gradOutShape.GetDim(i);
    }
    tailDimLength = gradOutShape.GetDim(gradOutDim - 1);
    auto temp = tilingContext->GetInputDesc(0);
    dataType = temp->GetDataType();
    blockSize = BLOCK / dataTypeSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpSegsumGradTiling::RunBigKernelTiling()
{
    ParseInputAttrs();
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;
    auto compileInfo = reinterpret_cast<const ExpSegsumGradCompileInfo*>(tilingContext->GetCompileInfo());
    uint32_t coreNumPlatform = compileInfo->coreNum;
    uint64_t ubSizePlatform = compileInfo->ubSize;
    GetNeedCoreNum(coreNumPlatform);
    GetTilingKey(ubSizePlatform);
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

void ExpSegsumGradTiling::GetTilingKey(uint64_t ubSizePlatform)
{
    int64_t eachTensorSize = (ubSizePlatform - REDUCE_SUM_SIZE) / TENSOR_NUM;
    int64_t maxSlideSizeUnalign = eachTensorSize / dataTypeSize;
    int64_t maxSlideSize = maxSlideSizeUnalign / blockSize * blockSize;

    int64_t rowNum = 1;
    int64_t colNum = maxSlideSize;
    int32_t blockSizeReal = BLOCK / GetDataTypeSize();

    if (tailDimLength <= maxSlideSize) {
        int64_t calNumAlign = CeilA2B(tailDimLength, blockSizeReal) * blockSizeReal;
        rowNum = maxSlideSize / calNumAlign;
        colNum = tailDimLength;
    }
    auto shape = ge::Shape({rowNum, colNum});
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetReduceSumMaxMinTmpSize(
        shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, true, false, maxValue, minValue);
    if (minValue > REDUCE_SUM_SIZE) {
        int64_t size = CeilA2B(CeilA2B(minValue - REDUCE_SUM_SIZE, TENSOR_NUM), BLOCK) * BLOCK;
        maxSlideSize = maxSlideSize - size / GetDataTypeSize();
    }
    if (tailDimLength <= maxSlideSize) {
        tilingKey = SMALL_SIZE_TILING_KEY;
    } else {
        tilingKey = COMMON_TILING_KEY;
    }

    int64_t tailDimLengthAlign = CeilA2B(tailDimLength, blockSizeReal) * blockSizeReal;
    slideSize = std::min(tailDimLength * tailDimLengthAlign, maxSlideSize);
}

void ExpSegsumGradTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    int64_t averageBatches = CeilA2B(batches, coreNumPlatform);
    needCoreNum = CeilA2B(batches, averageBatches);
    for (int64_t coreIndex = 0; coreIndex < needCoreNum; coreIndex++) {
        batchStart[coreIndex] = coreIndex * averageBatches;
        batchEnd[coreIndex] = std::min((coreIndex + 1) * averageBatches, batches);
    }
}

uint8_t ExpSegsumGradTiling::GetDataTypeSize()
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

template <typename T1, typename T2>
inline auto ExpSegsumGradTiling::CeilA2B(T1 a, T2 b) -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

void ExpSegsumGradTiling::FillTilingData()
{
    tilingData.set_batches(batches);
    tilingData.set_tailDimLength(tailDimLength);
    tilingData.set_batchStart(batchStart);
    tilingData.set_batchEnd(batchEnd);
    tilingData.set_slideSize(slideSize);
    tilingData.set_needCoreNum(needCoreNum);

    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(tilingKey);
    if (tilingKey == COMMON_TILING_KEY) {
        tilingContext->SetScheduleMode(SCHEDULE_MODE);
    }

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus Tiling4ExpSegsumGrad(gert::TilingContext* context)
{
    ExpSegsumGradTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepare4ExpSegsumGrad(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ExpSegsumGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->ubSize = ubSize;

    OP_CHECK_IF(
        compileInfo->coreNum <= 0,
        OP_LOGE(
            context->GetNodeName(), "ExpSegsumGrad GetHardwareInfo Failed, vectorCoreNum: %u", compileInfo->coreNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compileInfo->ubSize <= 0,
        OP_LOGE(
            context->GetNodeName(), "ExpSegsumGrad GetHardwareInfo Failed, ubSize: %lu", compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ExpSegsumGrad)
    .Tiling(Tiling4ExpSegsumGrad)
    .TilingParse<ExpSegsumGradCompileInfo>(TilingPrepare4ExpSegsumGrad);
} // namespace optiling