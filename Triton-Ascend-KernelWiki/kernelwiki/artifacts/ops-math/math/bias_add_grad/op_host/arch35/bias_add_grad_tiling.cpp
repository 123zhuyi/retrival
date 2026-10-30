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
 * \file bias_add_grad_tiling.cpp
 * \brief bias_add_grad
 */
#include <map>
#include "bias_add_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "bias_add_grad_tiling_arch35.h"
#include "log/log.h"
using namespace ge;

namespace optiling {

static ge::graphStatus Tiling4BiasAddGrad(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const BiasAddGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    return Tiling4BiasAddGradForAscendC(context);
}

static ge::graphStatus TilingPrepare4BiasAddGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the BiasAddGrad op.
IMPL_OP_OPTILING(BiasAddGrad).Tiling(Tiling4BiasAddGrad).TilingParse<BiasAddGradCompileInfo>(TilingPrepare4BiasAddGrad);
} // namespace optiling
