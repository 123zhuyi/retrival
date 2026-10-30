/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rint_tiling_arch35.cpp
 * \brief
 */

#include "rint_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "math/rint/op_kernel/arch35/rint_dag.h"
#include "math/rint/op_kernel/arch35/rint_tiling_struct.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "op_host/tiling_base_util.h"
#include "util/math_util.h"

using namespace AscendC;
using namespace ge;
using namespace RintNs;
using namespace RintOp;
using namespace Ops::Base;

namespace optiling {

ge::graphStatus RintTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "RintTiling CalcInputDtype enter.");

    auto IsSupportedDtype = [](ge::DataType dtype) -> bool {
        return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT;
    };

    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedDtype(this->inputDtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    tilingContext->GetNodeName(), "x", ToString(this->inputDtype).c_str(),
                    "The dtype of x must be within the range DT_FLOAT16, DT_BF16 and DT_FLOAT"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RintTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "RintTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    std::string errorMsg = "The dtype of y must be the same as " + ToString(this->inputDtype) + " of x";
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "y",
                                                      ToString(this->outputDtype).c_str(), errorMsg.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RintTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "RintTiling CheckShape enter.");
    auto xStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& inputXShape = EnsureNotScalar(xStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputXShape != outputYShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), "y", ToString(outputYShape).c_str(),
                                                      "The shape of y must be equal to the shape of x"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RintTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "RintTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);

    if (CalcInputDtype() == ge::GRAPH_FAILED || CalcOutputDtype() == ge::GRAPH_FAILED ||
        CheckShape() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);

    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->inputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        res = elewiseBaseTiling.DoTiling<RintDag<half>::OpDag>(*tiling);
    } else if (this->inputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        res = elewiseBaseTiling.DoTiling<RintDag<bfloat16_t>::OpDag>(*tiling);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        res = elewiseBaseTiling.DoTiling<RintDag<float>::OpDag>(*tiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(), "x", ToString(this->inputDtype).c_str(),
            "The dtype of x must be within the range DT_FLOAT16, DT_BF16 and DT_FLOAT");
        return ge::GRAPH_FAILED;
    }

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tiling_Key=%ld", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);

    return res;
}

ge::graphStatus TilingForRint(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext->GetNodeName(), "TilingForRint arch35 is running");
    auto compileInfo = tilingContext->GetCompileInfo<RintCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    RintTiling baseOpTiling(tilingContext);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForRint([[maybe_unused]] gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(Rint).Tiling(TilingForRint).TilingParse<RintCompileInfo>(TilingPrepareForRint);

} // namespace optiling
