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
 * \file atan_tiling_arch35.cpp
 * \brief atan_tiling source file
 */

#include <iostream>
#include <graph/utils/type_utils.h>

#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "math/atan/op_kernel/arch35/atan_dag.h"
#include "math/atan/op_kernel/arch35/atan_struct.h"
#include "atan_tiling_arch35.h"

using namespace ge;
using namespace AtanOp;

namespace optiling {
constexpr int64_t ASCEND_API_BUFFER = 122880;
const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;

ge::graphStatus AtanTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "input",
                                               ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
                                               "The dtype of input must be FLOAT16, BF16 or FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AtanTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "output",
                                               ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                               "The dtype of output must be FLOAT16, BF16 or FLOAT"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "input, output",
                                                ge::TypeUtils::DataTypeToSerialString(this->inputDtype) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
                                                "The dtypes of input and output must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AtanTiling::CheckShape()
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape != outputYShape, OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "inputX, outputY",
                                                                            (Ops::Base::ToString(inputXShape) + ", " +
                                                                             Ops::Base::ToString(outputYShape)).c_str(),
                                                                            "The shapes of input and output must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AtanTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "Enter TilingForAtan");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get AtanTiling from GE context failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus ret = ge::GRAPH_SUCCESS;

    if (inputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        ret = elewiseBaseTiling.DoTiling<AtanOp::AtanDag<float>::OpDag>(*tiling, ASCEND_API_BUFFER);
    } else if (inputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        ret = elewiseBaseTiling.DoTiling<AtanOp::AtanDag<half>::OpDag>(*tiling, ASCEND_API_BUFFER);
    } else if (inputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        ret = elewiseBaseTiling.DoTiling<AtanOp::AtanDag<bfloat16_t>::OpDag>(*tiling, ASCEND_API_BUFFER);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "inputDtype",
                                       ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
                                       "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    // set workspace/tilingkey/blockdim
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->scheMode), dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu.", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);

    return ret;
}

ge::graphStatus TilingForAtan(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4Atan rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    AtanTiling baseOpTiling(context);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForAtan([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Atan).Tiling(TilingForAtan).TilingParse<ElewiseCompileInfo>(TilingPrepareForAtan);

} // namespace optiling