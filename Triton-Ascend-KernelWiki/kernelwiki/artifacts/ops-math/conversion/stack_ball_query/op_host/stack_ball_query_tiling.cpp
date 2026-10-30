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
 * \file stack_ball_query.cpp
 * \brief
 */
#include "stack_ball_query_tiling.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"

using namespace ge;
namespace optiling {
constexpr int32_t INDEX_INPUT_XYZ = 0;
constexpr int32_t INDEX_INPUT_CENTER_XYZ = 1;
constexpr int32_t INDEX_INPUT_XYZ_BATCH_CNT = 2;
constexpr int32_t INDEX_OUTPUT_IDX = 0;
constexpr uint32_t WORKSPACE_16MB_SIZE = 16 * 1024 * 1024;

constexpr size_t MAX_RADIUS_IDX = 0;
constexpr size_t SAMPLE_NUM_IDX = 1;
constexpr int32_t FP32_MODE = 1;
constexpr int32_t FP16_MODE = 2;

static int32_t GetCeilInt(int32_t num1, int32_t num2)
{
    if (num2 != 0) {
        return (num1 + num2 - 1) / num2;
    }
    return 0;
}

class StackBallQueryTiling {
public:
    explicit StackBallQueryTiling(gert::TilingContext* context) : tilingContext(context){};

    void Init() const;

    ge::graphStatus RunKernelTiling();

    void CalRunningInfo(gert::TilingContext* context, const uint64_t actCoreNum);

    void TilingDataPrint() const;

private:
    StackBallQueryTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    int32_t batchSize;
    int32_t totalLengthCenterXyz;
    int32_t totalLengthXyz;
    int32_t totalIdxLength;
    int32_t coreNum;
    int32_t centerXyzPerCore;
    int32_t tailCenterXyzPerCore;
    float maxRadius;
    int32_t sampleNum;
};

void StackBallQueryTiling::Init() const
{
    OP_LOGD(tilingContext, "tiling initing.");
    auto dataType = tilingContext->GetInputTensor(INDEX_INPUT_XYZ)->GetDataType();
    tilingContext->SetTilingKey(FP32_MODE);
    if (dataType == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(FP32_MODE);
        OP_LOGD(tilingContext, "set tilingKey to FP32_MODE.");
    } else if (dataType == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(FP16_MODE);
        OP_LOGD(tilingContext, "set tilingKey to FP16_MODE.");
    }
    OP_LOGD(tilingContext, "tiling inited.");
}

void StackBallQueryTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "TilingDataPrint start.");
    OP_LOGD(tilingContext, "batchSize is %d.", this->batchSize);
    OP_LOGD(tilingContext, "totalLengthCenterXyz is %d.", this->totalLengthCenterXyz);
    OP_LOGD(tilingContext, "totalLengthXyz is %d.", this->totalLengthXyz);
    OP_LOGD(tilingContext, "totalIdxLength is %d.", this->totalIdxLength);
    OP_LOGD(tilingContext, "coreNum is %d.", this->coreNum);
    OP_LOGD(tilingContext, "centerXyzPerCore is %d.", this->centerXyzPerCore);
    OP_LOGD(tilingContext, "tailCenterXyzPerCore is %d.", this->tailCenterXyzPerCore);
    OP_LOGD(tilingContext, "sampleNum is %d.", this->sampleNum);
    OP_LOGD(tilingContext, "maxRadius is %f.", this->maxRadius);
    OP_LOGD(tilingContext, "TilingDataPrint end.");
}

void StackBallQueryTiling::CalRunningInfo(gert::TilingContext* context, const uint64_t actCoreNum)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context, "[CalRunningInfo] attrs is null."), return);
    const int64_t* sampleNumPtr = attrs->GetAttrPointer<int64_t>(SAMPLE_NUM_IDX);
    this->sampleNum = *sampleNumPtr;

    const float* maxRadiusPtr = attrs->GetAttrPointer<float>(MAX_RADIUS_IDX);
    this->maxRadius = *maxRadiusPtr;

    auto runtimeCenterXyzShapePtr = context->GetInputShape(INDEX_INPUT_CENTER_XYZ);
    OP_CHECK_IF(
        runtimeCenterXyzShapePtr == nullptr, OP_LOGE(context, "[CalRunningInfo] runtimeCenterXyzShapePtr is null."),
        return);
    const gert::Shape& centerXyzShape = runtimeCenterXyzShapePtr->GetStorageShape();

    auto runtimeXyzShapePtr = context->GetInputShape(INDEX_INPUT_XYZ);
    OP_CHECK_IF(
        runtimeXyzShapePtr == nullptr, OP_LOGE(context, "[CalRunningInfo] runtimeXyzShapePtr is null."), return);
    const gert::Shape& xyzShape = runtimeXyzShapePtr->GetStorageShape();

    auto runtimeXyzBatchCntShapePtr = context->GetInputShape(INDEX_INPUT_XYZ_BATCH_CNT);
    OP_CHECK_IF(
        runtimeXyzBatchCntShapePtr == nullptr, OP_LOGE(context, "[CalRunningInfo] runtimeXyzBatchCntShapePtr is null."),
        return);
    const gert::Shape& xyzBatchCntShape = runtimeXyzBatchCntShapePtr->GetStorageShape();

    this->batchSize = xyzBatchCntShape.GetDim(0);
    this->totalLengthCenterXyz = centerXyzShape.GetDim(0);
    this->totalLengthXyz = xyzShape.GetDim(1);
    this->totalIdxLength = totalLengthCenterXyz * this->sampleNum;

    if (static_cast<uint64_t>(this->totalLengthCenterXyz) <= actCoreNum) {
        this->coreNum = totalLengthCenterXyz;
    } else {
        this->coreNum = static_cast<int32_t>(actCoreNum);
    }

    this->centerXyzPerCore = GetCeilInt(this->totalLengthCenterXyz, this->coreNum);
    int32_t alignNum = 8;
    if (GetCeilInt(alignNum, this->sampleNum) > this->centerXyzPerCore) {
        this->centerXyzPerCore = GetCeilInt(alignNum, this->sampleNum);
    }

    this->tailCenterXyzPerCore = this->totalLengthCenterXyz % this->centerXyzPerCore;
    if (this->tailCenterXyzPerCore == 0) {
        this->coreNum = this->totalLengthCenterXyz / this->centerXyzPerCore;
    } else {
        this->coreNum = 1 + (this->totalLengthCenterXyz - this->tailCenterXyzPerCore) / this->centerXyzPerCore;
    }
}

ge::graphStatus StackBallQueryTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "RunKernelTiling start.");

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    const uint64_t actCoreNum = platformInfo.GetCoreNumAiv();

    CalRunningInfo(tilingContext, actCoreNum);

    tilingData.set_batchSize(this->batchSize);
    tilingData.set_totalLengthCenterXyz(this->totalLengthCenterXyz);
    tilingData.set_totalLengthXyz(this->totalLengthXyz);
    tilingData.set_totalIdxLength(this->totalIdxLength);
    tilingData.set_coreNum(this->coreNum);
    tilingData.set_centerXyzPerCore(this->centerXyzPerCore);
    tilingData.set_tailCenterXyzPerCore(this->tailCenterXyzPerCore);
    tilingData.set_maxRadius(this->maxRadius);
    tilingData.set_sampleNum(this->sampleNum);

    tilingContext->SetBlockDim(tilingData.get_coreNum());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();

    size_t sysWorkspaceSize = WORKSPACE_16MB_SIZE;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD(tilingContext, "RunKernelTiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingStackBallQuery(gert::TilingContext* context)
{
    StackBallQueryTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus TilingPrepare4StackBallQuery(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4StackBallQuery enter.");
    auto compileInfo = context->GetCompiledInfo<StackBallQueryCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicore_num = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(
        (compileInfo->aicore_num <= 0),
        OP_LOGE(
            context->GetNodeName(), "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->aicore_num)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ub_platform_byte_size = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ub_platform_byte_size <= 0),
        OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ub_platform_byte_size)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StackBallQuery)
    .Tiling(TilingStackBallQuery)
    .TilingParse<StackBallQueryCompileInfo>(TilingPrepare4StackBallQuery);

} // namespace optiling
