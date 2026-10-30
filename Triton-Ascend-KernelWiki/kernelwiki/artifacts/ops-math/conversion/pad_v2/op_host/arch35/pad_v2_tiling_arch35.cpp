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
 * \file pad_v2_tiling_arch35.cpp
 * \brief PadV2 tiling - 复用 PadV3 实现
 */
#include "pad_v2_tiling_arch35.h"
#include "conversion/pad_v3/op_host/arch35/pad_v3_tiling_arch35.h"
#include "log/log.h"

namespace optiling {

static ge::graphStatus Tiling4PadV2(gert::TilingContext* context)
{
    // 直接使用 PadV3 的 Tiling 类
    PadACTiling tilingObject(context);
    return tilingObject.DoTiling();
}

static ge::graphStatus TilingPrepare4PadV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4PadV2 entering.");
    return ge::GRAPH_SUCCESS;
}

// 注册 PadV2 的 tiling 接口
IMPL_OP_OPTILING(PadV2).Tiling(Tiling4PadV2).TilingParse<PadV3CompileInfo>(TilingPrepare4PadV2);

} // namespace optiling
