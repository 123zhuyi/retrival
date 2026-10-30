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
 * \file complex_tiling_arch35.cpp
 * \brief complex broadcast tiling implementation
 */
#include <graph/utils/type_utils.h>
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/complex_dag.h"
#include "../../op_kernel/arch35/complex_struct.h"
#include "complex_tiling_arch35.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t COMPLEX_COMMON_TILING_PRIORITY = 0;

ge::graphStatus ComplexTiling::GetShapeAttrsInfo() { return ge::GRAPH_SUCCESS; }

bool ComplexTiling::IsCapable() { return true; }

ge::graphStatus ComplexTiling::DoOpTiling()
{
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (outputDtype == ge::DT_COMPLEX64) {
        BroadcastBaseTiling<ComplexOp::ComplexBrcDag<complex64, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (outputDtype == ge::DT_COMPLEX32) {
        BroadcastBaseTiling<ComplexOp::ComplexBrcDag<complex32, half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "output", ge::TypeUtils::DataTypeToSerialString(outputDtype),
                                  "COMPLEX64, COMPLEX32");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus ComplexTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ComplexTiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus ComplexTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ComplexTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ComplexTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus TilingForComplex(gert::TilingContext* context)
{
    OP_LOGD("ComplexTiling", "Enter TilingForComplex");
    if (context == nullptr) {
        OP_LOGE("ComplexTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const ComplexCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc ComplexTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForComplex(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ComplexCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Complex).Tiling(TilingForComplex).TilingParse<ComplexCompileInfo>(TilingPrepareForComplex);

REGISTER_OPS_TILING_TEMPLATE(Complex, ComplexTiling, COMPLEX_COMMON_TILING_PRIORITY);
} // namespace optiling
