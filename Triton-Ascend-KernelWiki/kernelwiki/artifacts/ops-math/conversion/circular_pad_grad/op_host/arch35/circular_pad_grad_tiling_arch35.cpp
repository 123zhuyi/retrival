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
 * \file circular_pad_grad
 * \brief
 */

#include "conversion/pad_v3_grad/op_host/arch35/pad_v3_grad_tiling_arch35.h"
#include "log/log.h"

namespace optiling {
struct CircularPadGradCompileInfo {};

static ge::graphStatus CircularPadGradTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "CircularPadGradTiling running begin");
    PadV3GradACTiling tilingObject(context);
    tilingObject.isCircularPadGrad_ = true;
    return tilingObject.DoTiling();
}

static ge::graphStatus TilingPrepareForCircularPadGrad(gert::TilingParseContext* context)
{
    auto ci = context->GetCompiledInfo<CircularPadGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the CircularPadGrad op.
IMPL_OP_OPTILING(CircularPadGrad)
    .Tiling(CircularPadGradTiling)
    .TilingInputsDataDependency({1})
    .TilingParse<CircularPadGradCompileInfo>(TilingPrepareForCircularPadGrad);

} // namespace optiling
