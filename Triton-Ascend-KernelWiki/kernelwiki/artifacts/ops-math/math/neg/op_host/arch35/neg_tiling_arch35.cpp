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
 * \file neg_regbase_optiling.cc
 * \brief
 */

#include <iostream>
#include <graph/utils/type_utils.h>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/const_util.h"
#include "util/math_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "math/neg/op_kernel/arch35/neg_tiling_struct.h"
#include "math/neg/op_kernel/arch35/neg_dag.h"
#include "math/neg/op_kernel/arch35/neg_struct.h"

using namespace ge;
using namespace NegOp;
using namespace AscendC;
using namespace Ops::Base;

namespace optiling {

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

class NegTiling {
public:
    explicit NegTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    ge::graphStatus SetTilingData();
    NegTilingData* tiling = nullptr;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CheckOutputDtype();
    ge::graphStatus CheckOutputShape();

private:
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    uint32_t numBlocks = 0;
    int64_t outputSize = 0;
    gert::TilingContext* tilingContext;
    uint64_t tilingKey = 0;
};

ge::graphStatus NegTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NegTiling::CheckOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    OP_CHECK_IF(
        this->outputDtype != inputDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            tilingContext->GetNodeName(), "x, y",
            std::string(ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType())) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)),
            "The dtypes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NegTiling::CheckOutputShape()
{
    const auto inputDsc = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDsc);
    const auto outputDsc = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDsc);
    // get storage shape
    gert::Shape inputShape = inputDsc->GetStorageShape();
    gert::Shape outputShape = outputDsc->GetStorageShape();
    // check the input shape and output shape are the same
    OP_CHECK_IF(
        (inputShape != outputShape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            tilingContext->GetNodeName(), "x, y",
            (Ops::Base::ToString(inputShape) + ", " + Ops::Base::ToString(outputShape)).c_str(),
            "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NegTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    tiling = tilingContext->GetTilingData<NegTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    // 获取tiling计算所需的参数
    ge::graphStatus status = CalcOutputDtype();
    OP_CHECK_IF(
        status != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "Get output dtype failed"), return ge::GRAPH_FAILED);
    status = CheckOutputDtype();
    OP_CHECK_IF(
        status != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "CheckOutputDtype failed"), return ge::GRAPH_FAILED);
    status = CheckOutputShape();
    OP_CHECK_IF(
        status != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "CheckOutputShape failed"), return ge::GRAPH_FAILED);

    if (this->outputDtype == ge::DT_FLOAT16) {
        status = elewiseBaseTiling.DoTiling<NegDag::NegNoCast<half>::OpDag>(tiling->baseTiling);
        tilingKey = GET_TPL_TILING_KEY(tiling->baseTiling.scheMode, NEG_TPL_FP16);
    } else if (this->outputDtype == ge::DT_BF16) {
        status = elewiseBaseTiling.DoTiling<NegDag::NegNeedCast<bfloat16_t>::OpDag>(tiling->baseTiling);
        tilingKey = GET_TPL_TILING_KEY(tiling->baseTiling.scheMode, NEG_TPL_BF16);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        status = elewiseBaseTiling.DoTiling<NegDag::NegNoCast<float>::OpDag>(tiling->baseTiling);
        tilingKey = GET_TPL_TILING_KEY(tiling->baseTiling.scheMode, NEG_TPL_FP32);
    } else if (this->outputDtype == ge::DT_INT32) {
        status = elewiseBaseTiling.DoTiling<NegDag::NegNoCast<int32_t>::OpDag>(tiling->baseTiling);
        tilingKey = GET_TPL_TILING_KEY(tiling->baseTiling.scheMode, NEG_TPL_INT32);
    } else if (this->outputDtype == ge::DT_INT8) {
        status = elewiseBaseTiling.DoTiling<NegDag::NegNoCast<int8_t>::OpDag>(tiling->baseTiling);
        tilingKey = GET_TPL_TILING_KEY(tiling->baseTiling.scheMode, NEG_TPL_INT8);
    } else if (this->outputDtype == ge::DT_INT64) {
        status = elewiseBaseTiling.DoTiling<NegDag::NegNoCast<int64_t>::OpDag>(tiling->baseTiling);
        tilingKey = GET_TPL_TILING_KEY(tiling->baseTiling.scheMode, NEG_TPL_INT64);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            tilingContext->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
            "FLOAT16, BF16, FLOAT, INT32, INT8, INT64");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        status != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "ElewiseBaseTiling do tiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

ge::graphStatus NegTiling::SetTilingData()
{
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);
    size_t usrWorkspaceSize = 0;
    size_t sysWorkspaceSize = static_cast<size_t>(16 * 1024 * 1024);
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;
    OP_LOGD(tilingContext->GetNodeName(), "END Neg AscendC Tiling \n");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncNeg(gert::TilingContext* tilingContext)
{
    auto compileInfo = tilingContext->GetCompileInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    // 走新的模板tiling
    OP_LOGD(tilingContext->GetNodeName(), "START Neg AscendC Tiling \n");
    NegTiling NegTiling(tilingContext);
    return NegTiling.RunTiling();
}

ge::graphStatus TilingPrepareForNeg(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Neg).Tiling(TilingFuncNeg).TilingParse<ElewiseCompileInfo>(TilingPrepareForNeg);
} // namespace optiling
