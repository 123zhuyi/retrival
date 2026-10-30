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
 * \file fallback_quant_reduce_scatter.cpp
 * \brief 动态shape图回调aclnn
*/
#include "fallback/fallback.h"
#include "op_mc2.h"
#include "mc2_log.h"
 
namespace fallback {

const char* QuantReduceScatterInfo = "QuantReduceScatterFallback";

static ge::graphStatus QuantReduceScatterExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D(QuantReduceScatterInfo, "Start to fallback for quant_reduce_scatter.");
    OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);

    // 校验tensor
    const auto x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
    OPS_ERR_IF(x == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "x is null"), return ge::GRAPH_FAILED);
    const auto scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(1));
    OPS_ERR_IF(scales == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "scales is null"), return ge::GRAPH_FAILED);
    const auto output = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
    OPS_ERR_IF(output == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "output is null"), return ge::GRAPH_FAILED);

    // 校验attrs
    const auto attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "attrs is null"), return ge::GRAPH_FAILED);
    const char *group = attrs->GetStr(static_cast<size_t>(0));
    OPS_ERR_IF(group == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "group is null"), return ge::GRAPH_FAILED);
    const char *reduce_op = attrs->GetStr(static_cast<size_t>(1));
    OPS_ERR_IF(reduce_op == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "reduce_op is null"), return ge::GRAPH_FAILED);

    // 执行回调
    const auto ret = EXEC_OPAPI_CMD(aclnnQuantReduceScatter, x, scales, group, reduce_op, output);
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
               OPS_LOG_E(QuantReduceScatterInfo, "Aclnn api error code %d", ret),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantReduceScatter).OpExecuteFunc(QuantReduceScatterExecuteFunc);

} // namespace fallback
