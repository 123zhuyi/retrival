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
 * \file attention_to_ffn_gen_task.cpp
 * \brief
 */
#include <vector>
#include <set>
#include <string>

#include "op_mc2.h"
#include "platform/platform_info.h"

#ifdef BUILD_OPEN_PROJECT
#include "mc2_gen_task_ops_utils.h"
#include "mc2_moe_gen_task_ops_utils.h"
#include "graph/arg_desc_info.h"
#include "graph/kernel_launch_info.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#else
#include "ops_error.h"
#include "mc2_gen_task_moe.h"
#include "mc2_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#include "register/op_ext_gentask_registry.h"
#endif

namespace ops {
constexpr char AICPU_KFC_SERVER_NAME[] = "aicpu kfc server";
constexpr char KFC_STREAM_NAME[] = "kfc_stream";

#ifdef BUILD_OPEN_PROJECT
ge::Status AttentionToFFNCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = AICPU_KFC_SERVER_NAME;
    const ge::AscendString reuseKey = KFC_STREAM_NAME;
    
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status AttentionToFFNGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "MC2 Generate task start.");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(AttentionToFFN)
    .CalcOpParam(AttentionToFFNCalcParamFunc)
    .GenerateTask(AttentionToFFNGenTaskFunc);
#else // mc2 gen task utils
ge::Status AttentionToFFNCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = AICPU_KFC_SERVER_NAME;
    const ge::AscendString reuseKey = KFC_STREAM_NAME;
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status AttentionToFFNGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    OPS_LOG_I(context->GetNodeName(), "MC2 Generate task start.");
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallbackV2);
}

IMPL_OP_CT(AttentionToFFN)
    .CalcOpParam(AttentionToFFNCalcParamFunc)
    .GenerateTask(AttentionToFFNGenTaskFunc);
REGISTER_EXT_TASK_TYPE(AttentionToFFN, fe::ExtTaskType::kAicoreTask);
#endif
} // namespace ops