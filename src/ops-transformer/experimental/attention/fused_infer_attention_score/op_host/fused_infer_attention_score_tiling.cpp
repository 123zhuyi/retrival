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
 * \file fused_infer_attention_score_tiling.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "arch32/fused_infer_attention_score_tiling_v3.h"


using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus TilingFusedInferAttentionScore(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (RouteToFia(context)) {
        return TilingFusedInferAttentionScoreV3(context);
    }
    return ge::GRAPH_SUCCESS;
}

FIA_EXTERN_C ge::graphStatus DoOpTilingFusedInferAttentionScore(gert::TilingContext *context)
{
    printf("this is A2/A3 demo!\n");
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("FusedInferAttentionScore", "Tiling context is null."),
        return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);
    
    return TilingFusedInferAttentionScore(context);
}

extern "C" {
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context)
{

}
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingFusedInferAttentionScore(
    gert::TilingContext *context)
{
    return DoOpTilingFusedInferAttentionScore(context);
}
}
} // namespace optiling