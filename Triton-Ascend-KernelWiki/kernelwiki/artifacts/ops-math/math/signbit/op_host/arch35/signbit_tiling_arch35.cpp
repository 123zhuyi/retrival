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
 * \file signbit_tiling_arch35.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "math/signbit/op_kernel/arch35/signbit_float_dag.h"
#include "math/signbit/op_kernel/arch35/signbit_integral_dag.h"
#include "math/signbit/op_kernel/arch35/signbit_tiling_struct.h"
#include "math/signbit/op_kernel/arch35/signbit_tilingdata.h"
#include "signbit_tiling_arch35.h"

using namespace ge;
using namespace SignbitOp;

namespace optiling {
const size_t SYS_WORKSPACE_SIZE = 0;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT8, ge::DT_UINT8,
                                              ge::DT_INT32, ge::DT_INT64,   ge::DT_UINT64, ge::DT_BOOL, ge::DT_DOUBLE};
constexpr static uint64_t SIGNBIT_COMMON_TILING_PRIORITY = 0;

class SignbitTiling {
public:
    explicit SignbitTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape() const;

private:
    ge::DataType inputDtype;
    uint64_t dType = 0;
    gert::TilingContext* tilingContext;
    SignbitTilingData* tiling = nullptr;
};
ge::graphStatus SignbitTiling::CheckDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        std::find(DTYPE_LIST.begin(), DTYPE_LIST.end(), inputDtype) == DTYPE_LIST.end(),
        OP_LOGE(
            tilingContext->GetNodeName(),
            "input1's dtype must be bf16/fp16/fp32/int8/uint8/int32/int64/uint64/bool/double,but: %s",
            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SignbitTiling::CheckShape() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SignbitTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "SignbitTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CheckDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<SignbitTilingData>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get SignbitTiling from GE context failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (inputDtype == ge::DT_INT64) {
        dType = TPL_INT64;
        ret = elewiseBaseTiling.DoTiling<SignbitIntegralOp::SignbitIntegralCompute<int64_t>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_UINT64) {
        dType = TPL_UINT64;
        ret =
            elewiseBaseTiling.DoTiling<SignbitIntegralOp::SignbitIntegralCompute<uint64_t>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_INT32) {
        dType = TPL_INT32;
        ret = elewiseBaseTiling.DoTiling<SignbitIntegralOp::SignbitIntegralCompute<int32_t>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        ret = elewiseBaseTiling.DoTiling<SignbitFloatCompute<float>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        ret = elewiseBaseTiling.DoTiling<SignbitFloatCompute<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        ret = elewiseBaseTiling.DoTiling<SignbitFloatCompute<half>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_UINT8) {
        dType = TPL_UINT8;
        ret = elewiseBaseTiling.DoTiling<SignbitIntegralOp::SignbitIntegralCompute<uint8_t>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_INT8) {
        dType = TPL_INT8;
        ret = elewiseBaseTiling.DoTiling<SignbitIntegralOp::SignbitIntegralCompute<int8_t>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_BOOL) {
        dType = TPL_BOOL;
        ret = elewiseBaseTiling.DoTiling<SignbitIntegralOp::SignbitIntegralCompute<int8_t>::OpDag>(tiling->baseTiling);
    } else if (inputDtype == ge::DT_DOUBLE) {
        dType = TPL_DOUBLE;
        ret = elewiseBaseTiling.DoTiling<SignbitDoubleCompute<double>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(
            tilingContext->GetNodeName(),
            "input dtype is only support uint64, int64, int32, float16, bf16, float, uint8, int8, bool, double!");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYS_WORKSPACE_SIZE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->baseTiling.scheMode), dType);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Signbit(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4Signbit rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const SignbitCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    SignbitTiling SignbitTiling(context);
    return SignbitTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForSignbit(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SignbitCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(Signbit).Tiling(Tiling4Signbit).TilingParse<SignbitCompileInfo>(TilingPrepareForSignbit);
} // namespace optiling