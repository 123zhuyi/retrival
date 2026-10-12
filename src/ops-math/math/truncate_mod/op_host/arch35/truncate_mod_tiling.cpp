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
 * \file truncate_mod_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "math/mod/op_host/arch35/mod_tiling_arch35.h"

namespace optiling {
constexpr static uint64_t MOD_COMMON_TILING_PRIORITY = 0;

static ge::graphStatus TilingForTruncateMod(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForTruncateMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
REGISTER_OPS_TILING_TEMPLATE(TruncateMod, ModTiling, MOD_COMMON_TILING_PRIORITY);
IMPL_OP_OPTILING(TruncateMod)
    .Tiling(TilingForTruncateMod)
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForTruncateMod);
} // namespace optiling
