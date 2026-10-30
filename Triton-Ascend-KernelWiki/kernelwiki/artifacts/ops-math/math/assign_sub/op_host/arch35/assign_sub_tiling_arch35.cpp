/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "assign_sub_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "math/assign_sub/op_kernel/arch35/assign_sub_dag.h"
#include "op_host/tiling_base_util.h"

using namespace ge;

namespace optiling {
const uint64_t ASSIGN_SUB_KEY_FP16 = 101UL;
const uint64_t ASSIGN_SUB_KEY_BF16 = 102UL;
const uint64_t ASSIGN_SUB_KEY_FP32 = 103UL;
const uint64_t ASSIGN_SUB_KEY_INT8 = 104UL;
const uint64_t ASSIGN_SUB_KEY_INT32 = 105UL;
const uint64_t ASSIGN_SUB_KEY_INT64 = 106UL;
const uint64_t ASSIGN_SUB_KEY_UINT8 = 107UL;
const size_t SYS_WORKSPACE_SIZE = 16777216;
const uint32_t DIM_NUM_9 = 9;

void AssignSubTiling::SetTilingData()
{
    if (this->outputDtype == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(ASSIGN_SUB_KEY_FP16);
    } else if (this->outputDtype == ge::DT_BF16) {
        tilingContext->SetTilingKey(ASSIGN_SUB_KEY_BF16);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(ASSIGN_SUB_KEY_FP32);
    } else if (this->outputDtype == ge::DT_INT8) {
        tilingContext->SetTilingKey(ASSIGN_SUB_KEY_INT8);
    } else if (this->outputDtype == ge::DT_INT32) {
        tilingContext->SetTilingKey(ASSIGN_SUB_KEY_INT32);
    } else if (this->outputDtype == ge::DT_INT64) {
        tilingContext->SetTilingKey(ASSIGN_SUB_KEY_INT64);
    } else if (this->outputDtype == ge::DT_UINT8) {
        tilingContext->SetTilingKey(ASSIGN_SUB_KEY_UINT8);
    }

    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
}

ge::graphStatus AssignSubTiling::CheckDtype()
{
    auto varDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, varDesc);
    auto valueDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, valueDesc);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    auto varDtype = varDesc->GetDataType();
    auto valueDtype = valueDesc->GetDataType();
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        varDtype != valueDtype || varDtype != this->outputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "var, value, output",
                                                ge::TypeUtils::DataTypeToSerialString(varDtype) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(valueDtype) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                                "The dtypes of var, value and output must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssignSubTiling::CheckShape() const
{
    auto varStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, varStorageShape);
    auto valueStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, valueStorageShape);
    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& varShape = Ops::Base::EnsureNotScalar(varStorageShape->GetStorageShape());
    const gert::Shape& valueShape = Ops::Base::EnsureNotScalar(valueStorageShape->GetStorageShape());
    const gert::Shape& outputShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        varShape != valueShape || varShape != outputShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "var, value, output",
                                                (Ops::Base::ToString(varShape) + ", " + Ops::Base::ToString(valueShape) +
                                                 ", " + Ops::Base::ToString(outputShape)).c_str(),
                                                "The shapes of var, value and output must be the same"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssignSubTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "AssignSubTiling RunTiling enter.");
    Ops::Base::ElewiseBaseTiling eleBaseTiling(tilingContext);

    tiling = tilingContext->GetTilingData<AssignSubTilingData>();
    OP_CHECK_IF(
        CheckDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);
    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        ret = eleBaseTiling.DoTiling<AssignSubOp<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        ret = eleBaseTiling.DoTiling<AssignSubOp<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        ret = eleBaseTiling.DoTiling<AssignSubOp<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT8) {
        ret = eleBaseTiling.DoTiling<AssignSubOp<int8_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        ret = eleBaseTiling.DoTiling<AssignSubOp<int32_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT64) {
        ret = eleBaseTiling.DoTiling<AssignSubOp<int64_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_UINT8) {
        ret = eleBaseTiling.DoTiling<AssignSubOp<uint8_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "output",
                                       ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                       "FLOAT16, BF16, FLOAT, INT8, INT32, INT64, UINT8");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "AssignSubTiling failed"), return ge::GRAPH_FAILED);
    SetTilingData();
    size_t usrWorkspaceSize = 0;
    size_t sysWorkspaceSize = SYS_WORKSPACE_SIZE;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AssignSub(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4AssignSub rt2.0 is running.");
    auto compileInfo = context->GetCompileInfo<AssignSubCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    AssignSubTiling assignSubTiling(context);
    return assignSubTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForAssignSub(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AssignSubCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AssignSub)
    .Tiling(Tiling4AssignSub)
    .TilingParse<AssignSubCompileInfo>(
        TilingPrepareForAssignSub); // ASC自动生成的aclnn接口TilingParse注册的函数不会被调用，需要给ASC提需求解决
} // namespace optiling
