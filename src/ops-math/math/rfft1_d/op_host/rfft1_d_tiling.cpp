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
 * \file rfft1d_tiling.cc
 * \brief
 */

#include "rfft1_d_tiling.h"
#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_base_class.h"
#include "op_host/math_tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

static ge::graphStatus Tiling4Rfft1D(gert::TilingContext* context)
{
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4Rfft1D(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<Rfft1DCompileInfo>();
    OP_CHECK_IF(
        compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Rfft1D).Tiling(Tiling4Rfft1D).TilingParse<Rfft1DCompileInfo>(TilingPrepare4Rfft1D);

} // namespace optiling
