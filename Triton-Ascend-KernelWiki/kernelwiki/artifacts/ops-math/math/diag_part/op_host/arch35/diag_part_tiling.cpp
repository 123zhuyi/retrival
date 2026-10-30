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
 * \file diag_part.cc
 * \brief
 */
#include <cmath>
#include "diag_part_tiling_arch35.h"

using namespace std;

namespace optiling {

ge::graphStatus TilingPrepareDiagForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepareDiagForAscendC");
    auto ci = context->GetCompiledInfo<DiagCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (ci->core_num <= 0),
        OP_LOGE(context->GetNodeName(), "Diag Op GetHardwareInfo Failed, coreNum:%ld.", ci->core_num),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (ci->ub_size <= 0), OP_LOGE(context->GetNodeName(), "Diag Op GetHardwareInfo Failed, ubSize:%ld.", ci->ub_size),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Diag Op get coreNum:%ld, ubSize:%ld.", ci->core_num, ci->ub_size);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4Diag(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<DiagCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    TilingPrepareDiagForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4DiagPart(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4DiagPart running begin");
    const DiagCompileInfo* compile_info = reinterpret_cast<const DiagCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    return DiagPartTilingForAscendC(context);
}

// register tiling interface of the DiagTiling op.
IMPL_OP_OPTILING(DiagPart).Tiling(Tiling4DiagPart).TilingParse<DiagCompileInfo>(TilingPrepare4Diag);
} // namespace optiling