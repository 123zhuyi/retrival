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
 * \file concat_d_tiling_arch35.cpp
 * \brief
 */

#include "concat_d_tiling_arch35.h"
#include "conversion/concat/op_host/arch35/concat_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {

static ge::graphStatus Tiling4ConcatD(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("ConcatDTiling", "tiling context is nullptr"), return ge::GRAPH_FAILED);
    auto compile_info = context->GetCompileInfo<ConcatDCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "Tiling4ConcatD running begin");
    return TilingCommon(context, 0, -1);
}

static ge::graphStatus TilingPrepare4ConcatD(gert::TilingParseContext* context)
{
    return TilingPrepareForConcat(context);
}

// register tiling interface of the ConcatD op.
IMPL_OP_OPTILING(ConcatD).Tiling(Tiling4ConcatD).TilingParse<ConcatDCompileInfo>(TilingPrepare4ConcatD);
} // namespace optiling
