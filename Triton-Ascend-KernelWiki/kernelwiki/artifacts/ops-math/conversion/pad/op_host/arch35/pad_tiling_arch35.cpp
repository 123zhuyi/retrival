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
 * \file pad_tiling_arch35.cpp
 * \brief
 */
#include "pad_tiling_arch35.h"
#include "conversion/pad_v3/op_host/arch35/pad_v3_tiling_arch35.h"
#include "log/log.h"

namespace optiling {

static ge::graphStatus Tiling4Pad(gert::TilingContext* context)
{
    PadACTiling tilingObject(context);
    return tilingObject.DoTiling();
}

static ge::graphStatus TilingPrepare4Pad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPreparePad entering.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Pad op.
IMPL_OP_OPTILING(Pad).Tiling(Tiling4Pad).TilingParse<PadCompileInfo>(TilingPrepare4Pad);
} // namespace optiling
