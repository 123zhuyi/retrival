/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file one_hot_tiling.cpp
 * \brief
 */
#include "one_hot_tiling.h"
#include "one_hot_tiling_arch35.h"
#include "op_host/util/const_util.h"

using namespace ge;

namespace optiling {
static ge::graphStatus OneHotTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "OneHotTiling is running");
    const OneHotCompileInfo* compile_info = reinterpret_cast<const OneHotCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    return OneHotTilingForAscendC(context);
}

static ge::graphStatus TilingPrepare4OneHotForAscendc(gert::TilingParseContext* context, OneHotCompileInfo* compileInfo)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepare4OneHotForAscendc.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->core_num <= 0), OP_LOGE(context->GetNodeName(), "core num invalid."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ub_size <= 0), OP_LOGE(context->GetNodeName(), "ub size invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4OneHot(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForOneHot running.");
    auto compile_info = context->GetCompiledInfo<OneHotCompileInfo>();
    return TilingPrepare4OneHotForAscendc(context, compile_info);
}

IMPL_OP_OPTILING(OneHot)
    .Tiling(OneHotTiling)
    .TilingInputsDataDependency({ONEHOT_INPUT_DEPENDENCY_IDX})
    .TilingParse<OneHotCompileInfo>(TilingPrepare4OneHot);
} // namespace optiling