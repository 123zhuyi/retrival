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
 * \file drop_out_v3_grad_tiling.cpp
 * \brief
 */
#include "drop_out_v3_grad_tiling.h"
#include "drop_out_v3_grad_tiling_arch35.h"

namespace optiling {
// min num
const int64_t CORE_MINEST_NUM = 256;
// ub tiling: mask = grad_y / 8
const int64_t EIGHT_NUM = 8;
// ub tiling for float16 and float32: grad_y + grad_y / 8
const int64_t NINE_NUM = 9;
// ub tiling for bfloat16: grad_y + 2*grad_y + grad_y / 8
const int64_t TWENTY_FIVE_NUM = 25;

ge::graphStatus DropOutV3GradTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "DropOutV3GradTiling running begin");
    auto compileInfo = context->GetCompileInfo<DropOutV3GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return DropOutV3GradTilingForAscendC(context);
}

ge::graphStatus TilingPrepareDropOutV3GradForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareDropOutV3GradForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<DropOutV3GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Invalid ub size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDropOutV3Grad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForDropOutV3Grad running begin");
    auto compileInfo = context->GetCompiledInfo<DropOutV3GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return TilingPrepareDropOutV3GradForAscendC(context);
}

// register tiling interface of the DropOutV3Grad op.
IMPL_OP_OPTILING(DropOutV3Grad)
    .Tiling(DropOutV3GradTilingFunc)
    .TilingParse<DropOutV3GradCompileInfo>(TilingPrepareForDropOutV3Grad);
} // namespace optiling
