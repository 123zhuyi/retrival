/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_graph/op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {
using namespace ge;
using namespace gert;
static const size_t GRAD_Y_INDEX = 0;
static const size_t GROUP_IDX_INDEX = 1;

static graphStatus GroupedBiasAddGradHostExecFunc(OpExecuteContext* host_api_ctx)
{
    OP_LOGD("aclnnFallback", "GroupedBiasAddGrad fallback begin");
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto grad_y = host_api_ctx->GetInputTensor(GRAD_Y_INDEX);
    OP_CHECK_IF(grad_y == nullptr, OP_LOGE("aclnnfallback", "grad_y is null"), return GRAPH_FAILED);

    auto group_idx = host_api_ctx->GetOptionalInputTensor(GROUP_IDX_INDEX);

    auto output = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(output == nullptr, OP_LOGE("aclnnfallback", "output is null"), return GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return ge::GRAPH_FAILED);

    const int64_t* groupIdxType = attrs->GetAttrPointer<int64_t>(0);

    auto api_ret = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnGroupedBiasAddGradV2, grad_y, group_idx,
                                               *groupIdxType, output);

    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "api_ret failed:%u", api_ret), return GRAPH_FAILED);

    OP_LOGD("aclnnFallback", "GroupedBiasAddGrad fallback end");
    return GRAPH_SUCCESS;
}

IMPL_OP(GroupedBiasAddGrad).OpExecuteFunc(GroupedBiasAddGradHostExecFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
