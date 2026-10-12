/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stack_ball_query_tiling.cpp
 * \brief
 */
#include "stack_ball_query_tiling.h"
#include <cstdint>
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
constexpr int32_t INDEX_INPUT_CENTER_XYZ_BATCH_CNT = 3;
constexpr int32_t INDEX_OUTPUT_IDX = 0;
constexpr uint32_t WORKSPACE_16MB_SIZE = 16 * 1024 * 1024;

constexpr size_t MAX_RADIUS_IDX = 0;
constexpr size_t SAMPLE_NUM_IDX = 1;
constexpr int32_t FP32_MODE = 1;
constexpr int32_t FP16_MODE = 2;
constexpr int32_t FP32_INT64_MODE = 3;
constexpr int32_t FP16_INT64_MODE = 4;

static int32_t GetCeilInt(int32_t num1, int32_t num2)
{
    if (num2 != 0) {
        return (num1 + num2 - 1) / num2;
    }
    return 0;
}

// ========== ValidateDtype ==========
static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    auto opName = context->GetNodeName();

    auto xyzDesc = context->GetInputDesc(INDEX_INPUT_XYZ);
    OP_CHECK_NULL_WITH_CONTEXT(context, xyzDesc);
    auto xyzDtype = xyzDesc->GetDataType();
    OP_CHECK_IF(xyzDtype != ge::DT_FLOAT16 && xyzDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(opName, "xyz", std::to_string(static_cast<int32_t>(xyzDtype)).c_str(),
                                          "float16/float32"),
                return ge::GRAPH_FAILED);

    auto centerXyzDesc = context->GetInputDesc(INDEX_INPUT_CENTER_XYZ);
    OP_CHECK_NULL_WITH_CONTEXT(context, centerXyzDesc);
    auto centerXyzDtype = centerXyzDesc->GetDataType();
    OP_CHECK_IF(centerXyzDtype != xyzDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "xyz and center_xyz",
                                                       (std::to_string(static_cast<int32_t>(xyzDtype)) + " vs " +
                                                        std::to_string(static_cast<int32_t>(centerXyzDtype)))
                                                           .c_str(),
                                                       "center_xyz dtype must equal xyz dtype"),
                return ge::GRAPH_FAILED);

    auto xyzBatchCntDesc = context->GetInputDesc(INDEX_INPUT_XYZ_BATCH_CNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, xyzBatchCntDesc);
    auto xyzBatchCntDtype = xyzBatchCntDesc->GetDataType();
    OP_CHECK_IF(
        xyzBatchCntDtype != ge::DT_INT32 && xyzBatchCntDtype != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(opName, "xyz_batch_cnt",
                                  std::to_string(static_cast<int32_t>(xyzBatchCntDtype)).c_str(), "int32/int64"),
        return ge::GRAPH_FAILED);

    auto centerXyzBatchCntDesc = context->GetInputDesc(INDEX_INPUT_CENTER_XYZ_BATCH_CNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, centerXyzBatchCntDesc);
    auto centerXyzBatchCntDtype = centerXyzBatchCntDesc->GetDataType();
    OP_CHECK_IF(
        centerXyzBatchCntDtype != xyzBatchCntDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "xyz_batch_cnt and center_xyz_batch_cnt",
                                               (std::to_string(static_cast<int32_t>(xyzBatchCntDtype)) + " vs " +
                                                std::to_string(static_cast<int32_t>(centerXyzBatchCntDtype)))
                                                   .c_str(),
                                               "center_xyz_batch_cnt dtype must equal xyz_batch_cnt dtype"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ========== ValidateShape ==========
static ge::graphStatus ValidateShape(gert::TilingContext* context)
{
    auto opName = context->GetNodeName();

    auto xyzShape = context->GetInputShape(INDEX_INPUT_XYZ);
    OP_CHECK_NULL_WITH_CONTEXT(context, xyzShape);
    OP_CHECK_IF(xyzShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName, "xyz", (std::to_string(xyzShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);

    // xyz dim[0] 必须为 3（kernel 硬编码 XYZ_NUM=3，xyz 布局 [3, N]）
    OP_CHECK_IF(xyzShape->GetStorageShape().GetDim(0) != 3,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName, "xyz", ("xyz dim[0]=" + std::to_string(xyzShape->GetStorageShape().GetDim(0))).c_str(),
                    "xyz dim[0] must be 3 (x/y/z coordinates)"),
                return ge::GRAPH_FAILED);

    auto centerXyzShape = context->GetInputShape(INDEX_INPUT_CENTER_XYZ);
    OP_CHECK_NULL_WITH_CONTEXT(context, centerXyzShape);
    OP_CHECK_IF(
        centerXyzShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "center_xyz", (std::to_string(centerXyzShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    // center_xyz dim[1] 必须为 3（kernel 硬编码 XYZ_NUM=3）
    OP_CHECK_IF(centerXyzShape->GetStorageShape().GetDim(1) != 3,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName, "center_xyz",
                    ("center_xyz dim[1]=" + std::to_string(centerXyzShape->GetStorageShape().GetDim(1))).c_str(),
                    "center_xyz dim[1] must be 3 (x/y/z coordinates)"),
                return ge::GRAPH_FAILED);

    auto xyzBatchCntShape = context->GetInputShape(INDEX_INPUT_XYZ_BATCH_CNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, xyzBatchCntShape);
    OP_CHECK_IF(xyzBatchCntShape->GetStorageShape().GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName, "xyz_batch_cnt",
                    (std::to_string(xyzBatchCntShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);

    auto centerXyzBatchCntShape = context->GetInputShape(INDEX_INPUT_CENTER_XYZ_BATCH_CNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, centerXyzBatchCntShape);
    OP_CHECK_IF(centerXyzBatchCntShape->GetStorageShape().GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName, "center_xyz_batch_cnt",
                    (std::to_string(centerXyzBatchCntShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);

    // batch count 长度必须一致
    int64_t xyzBatchB = xyzBatchCntShape->GetStorageShape().GetDim(0);
    int64_t centerXyzBatchB = centerXyzBatchCntShape->GetStorageShape().GetDim(0);
    std::string batchMsg = "xyz_batch_cnt dim0=" + std::to_string(xyzBatchB) +
                           ", center_xyz_batch_cnt dim0=" + std::to_string(centerXyzBatchB);
    OP_CHECK_IF(
        xyzBatchB != centerXyzBatchB,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, "xyz_batch_cnt and center_xyz_batch_cnt", batchMsg.c_str(),
                                               "xyz_batch_cnt dim0 must equal center_xyz_batch_cnt dim0"),
        return ge::GRAPH_FAILED);

    // 溢出校验：TilingData 字段为 int32_t，shape 维度值和乘积不得溢出
    auto xyzDim0 = xyzShape->GetStorageShape().GetDim(0);
    auto xyzDim1 = xyzShape->GetStorageShape().GetDim(1);
    auto centerXyzDim0 = centerXyzShape->GetStorageShape().GetDim(0);
    OP_CHECK_IF(xyzDim0 > INT32_MAX || xyzDim1 > INT32_MAX || centerXyzDim0 > INT32_MAX || xyzBatchB > INT32_MAX,
                OP_LOGE(opName, "shape dim exceeds INT32_MAX: xyz=[%ld, %ld], centerXyz=[%ld, ...], batchSize=%ld",
                        xyzDim0, xyzDim1, centerXyzDim0, xyzBatchB),
                return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* sampleNumPtr = attrs->GetAttrPointer<int64_t>(SAMPLE_NUM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampleNumPtr);
    int64_t sampleNum = *sampleNumPtr;
    int64_t totalIdxLength = centerXyzDim0 * sampleNum;
    OP_CHECK_IF(totalIdxLength > INT32_MAX,
                OP_LOGE(opName, "totalIdxLength overflow: centerXyz=%ld * sampleNum=%ld = %ld", centerXyzDim0,
                        sampleNum, totalIdxLength),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ========== ValidateAttrs ==========
static ge::graphStatus ValidateAttrs(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* sampleNumPtr = attrs->GetAttrPointer<int64_t>(SAMPLE_NUM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampleNumPtr);
    int64_t sampleNum = *sampleNumPtr;
    OP_CHECK_IF(sampleNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sample_num",
                                                      ("sample_num=" + std::to_string(sampleNum)).c_str(),
                                                      "sample_num must be > 0"),
                return ge::GRAPH_FAILED);

    const float* maxRadiusPtr = attrs->GetAttrPointer<float>(MAX_RADIUS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxRadiusPtr);
    float maxRadius = *maxRadiusPtr;
    OP_CHECK_IF(maxRadius <= 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "max_radius",
                                                      ("max_radius=" + std::to_string(maxRadius)).c_str(),
                                                      "max_radius must be > 0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ========== ValidateInputs ==========
static ge::graphStatus ValidateInputs(gert::TilingContext* context)
{
    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAttrs(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateAttrs failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

class StackBallQueryTiling {
public:
    explicit StackBallQueryTiling(gert::TilingContext* context) : tilingContext(context) {};

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
    auto dataType = tilingContext->GetInputDesc(INDEX_INPUT_XYZ)->GetDataType();
    auto cntDataType = tilingContext->GetInputDesc(INDEX_INPUT_XYZ_BATCH_CNT)->GetDataType();
    tilingContext->SetTilingKey(FP32_MODE);
    if (dataType == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(cntDataType == ge::DT_INT64 ? FP32_INT64_MODE : FP32_MODE);
        OP_LOGD(tilingContext, "set tilingKey to FP32_MODE.");
    } else if (dataType == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(cntDataType == ge::DT_INT64 ? FP16_INT64_MODE : FP16_MODE);
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
    this->sampleNum = static_cast<int32_t>(*sampleNumPtr);

    const float* maxRadiusPtr = attrs->GetAttrPointer<float>(MAX_RADIUS_IDX);
    this->maxRadius = *maxRadiusPtr;

    auto runtimeCenterXyzShapePtr = context->GetInputShape(INDEX_INPUT_CENTER_XYZ);
    OP_CHECK_IF(runtimeCenterXyzShapePtr == nullptr,
                OP_LOGE(context, "[CalRunningInfo] runtimeCenterXyzShapePtr is null."), return);
    const gert::Shape& centerXyzShape = runtimeCenterXyzShapePtr->GetStorageShape();

    auto runtimeXyzShapePtr = context->GetInputShape(INDEX_INPUT_XYZ);
    OP_CHECK_IF(runtimeXyzShapePtr == nullptr, OP_LOGE(context, "[CalRunningInfo] runtimeXyzShapePtr is null."),
                return);
    const gert::Shape& xyzShape = runtimeXyzShapePtr->GetStorageShape();

    auto runtimeXyzBatchCntShapePtr = context->GetInputShape(INDEX_INPUT_XYZ_BATCH_CNT);
    OP_CHECK_IF(runtimeXyzBatchCntShapePtr == nullptr,
                OP_LOGE(context, "[CalRunningInfo] runtimeXyzBatchCntShapePtr is null."), return);
    const gert::Shape& xyzBatchCntShape = runtimeXyzBatchCntShapePtr->GetStorageShape();

    int64_t batchSizeInt64 = xyzBatchCntShape.GetDim(0);
    int64_t totalLengthCenterXyzInt64 = centerXyzShape.GetDim(0);
    int64_t totalLengthXyzInt64 = xyzShape.GetDim(1);

    this->batchSize = static_cast<int32_t>(batchSizeInt64);
    this->totalLengthCenterXyz = static_cast<int32_t>(totalLengthCenterXyzInt64);
    this->totalLengthXyz = static_cast<int32_t>(totalLengthXyzInt64);
    this->totalIdxLength = this->totalLengthCenterXyz * this->sampleNum;

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

    // 1. 校验输入
    OP_CHECK_IF(ValidateInputs(tilingContext) != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "ValidateInputs failed"),
                return ge::GRAPH_FAILED);

    // 2. 设置 TilingKey
    Init();

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
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
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
    OP_CHECK_IF((compileInfo->aicore_num <= 0),
                OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %u",
                        static_cast<uint32_t>(compileInfo->aicore_num)),
                return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ub_platform_byte_size = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ub_platform_byte_size <= 0),
                OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u",
                        static_cast<uint32_t>(compileInfo->ub_platform_byte_size)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StackBallQuery)
    .Tiling(TilingStackBallQuery)
    .TilingParse<StackBallQueryCompileInfo>(TilingPrepare4StackBallQuery);

} // namespace optiling
