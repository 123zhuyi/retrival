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
 * \file mirror_pad.cc
 * \brief
 */

#include "conversion/pad_v3/op_host/arch35/pad_v3_tiling_arch35.h"
#include "log/log.h"

namespace optiling {
struct MirrorPadCompileInfo {};

static ge::graphStatus MirrorPadTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "MirrorPadTiling running begin");
    PadACTiling tilingObject(context);
    tilingObject.isPadV3_ = true;
    tilingObject.isMirrorPad_ = true;
    return tilingObject.DoTiling();
}

static ge::graphStatus TilingPrepareForMirrorPad(gert::TilingParseContext* context)
{
    auto ci = context->GetCompiledInfo<MirrorPadCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the MirrorPad op.
IMPL_OP_OPTILING(MirrorPad)
    .Tiling(MirrorPadTiling)
    .TilingInputsDataDependency({1})
    .TilingParse<MirrorPadCompileInfo>(TilingPrepareForMirrorPad);

} // namespace optiling
