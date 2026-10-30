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
 * \file concat_v2_tiling_arch35.cpp
 * \brief concat_v2 tiling for ascendC impl
 */

#include "concat_v2_tiling_arch35.h"
#include "log/log.h"

namespace optiling {

static ge::graphStatus Tiling4ConcatV2ForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ConcatV2ForAscendC running begin");
    return optiling::TilingCommon(context, 0, 1);
}

IMPL_OP_OPTILING(ConcatV2)
    .TilingInputsDataDependency({1})
    .Tiling(Tiling4ConcatV2ForAscendC)
    .TilingParse<ConcatDCompileInfo>(TilingPrepareForConcat);

} // namespace optiling