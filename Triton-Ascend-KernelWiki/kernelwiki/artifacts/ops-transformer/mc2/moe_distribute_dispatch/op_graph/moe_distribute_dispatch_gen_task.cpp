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
 * \file moe_distribute_dispatch_gen_task.cpp
 * \brief
 */
#include <vector>

#ifdef BUILD_OPEN_PROJECT
#include "mc2_gen_task_ops_utils.h"
#include "mc2_moe_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#else
#include "ops_error.h"
#include "mc2_gen_task_moe.h"
#include "mc2_gen_task_utils.h"
#include "mc2_a5_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#include "register/op_ext_gentask_registry.h"
#endif

namespace ops {
#ifdef BUILD_OPEN_PROJECT
ge::Status MoeDistributeDispatchCalcParamFunc(gert::ExeResGenerationContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Do general calc param");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status MoeDistributeDispatchGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (Mc2GenTaskOpsUtils::IsTargetPlatformSocVersion(nodeName, PLATFORM_A2)) {
        OPS_LOG_D(context->GetNodeName(), "Do A2 gen task");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    OPS_LOG_D(context->GetNodeName(), "Do A3/A5 gen task");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(MoeDistributeDispatch)
    .CalcOpParam(MoeDistributeDispatchCalcParamFunc)
    .GenerateTask(MoeDistributeDispatchGenTaskFunc);
#else // mc2 gen task utils
ge::Status MoeDistributeDispatchCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MoeDistributeDispatchGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (Mc2A5GenTaskUtils::IsTargetPlatformSocVersion(nodeName, PLATFORM_A2)) {
        return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallback);
    }
    OPS_LOG_D(context->GetNodeName(), "Do MTE gen task.");
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallbackV2);
}

IMPL_OP_CT(MoeDistributeDispatch)
    .CalcOpParam(MoeDistributeDispatchCalcParamFunc)
    .GenerateTask(MoeDistributeDispatchGenTaskFunc);
REGISTER_EXT_TASK_TYPE(MoeDistributeDispatch, fe::ExtTaskType::kAicoreTask);
#endif
} // namespace ops
