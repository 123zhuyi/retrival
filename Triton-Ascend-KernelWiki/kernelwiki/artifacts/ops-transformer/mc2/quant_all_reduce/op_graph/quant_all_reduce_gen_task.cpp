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
 * \file quant_all_reduce_gen_task.cpp
 * \brief 静态shape图下沉实现
 */
#include <vector>
#include <platform/platform_info.h>
#include "op_mc2.h"

#ifdef BUILD_OPEN_PROJECT
#include "register/op_impl_registry.h"
#include "mc2_gen_task_ops_utils.h"
#include "mc2_gen_task_ops_utils_arch35.h"
#include "mc2_moe_gen_task_ops_utils.h"
#include "mc2_log.h"
#else
#include "ops_error.h"
#include "register/op_ext_gentask_registry.h"
#include "register/op_ct_impl_registry.h"
#include "mc2_gen_task_utils.h"
#include "mc2_gen_task_moe.h"
#include "mc2_a5_gen_task_utils.h"
#endif

namespace ops {

#ifdef BUILD_OPEN_PROJECT

static ge::Status QuantAllReduceCalcOpParam(gert::ExeResGenerationContext *context)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 MTE CalcParam in QuantAllReduce BUILD_OPEN_PROJECT");
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "mte server", "mte_stream");
    }
    OPS_LOG_D(context->GetNodeName(), "Do general CalcParam in QuantAllReduce BUILD_OPEN_PROJECT");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status QuantAllReduceGenTask(const gert::ExeResGenerationContext *context,
                                        std::vector<std::vector<uint8_t>> &tasks)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do MTE general GenTask in QuantAllReduce BUILD_OPEN_PROJECT");
        // 这里调用moe的接口
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
    }
    OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask in QuantAllReduce BUILD_OPEN_PROJECT");
    return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
}

IMPL_OP(QuantAllReduce)
    .CalcOpParam(QuantAllReduceCalcOpParam)
    .GenerateTask(QuantAllReduceGenTask);

#else

static ge::Status QuantAllReduceCalcOpParam(gert::ExeResGenerationContext *context)
{
    if (Mc2A5GenTaskUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 MTE CalcParam in QuantAllReduce");
        return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "mte server", "mte_stream");
    }
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

static ge::Status QuantAllReduceGenTask(const gert::ExeResGenerationContext *context,
                                        std::vector<std::vector<uint8_t>> &tasks)
{
    if (Mc2A5GenTaskUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do MTE general GenTask in QuantAllReduce");
        // 这里调用moe的接口
        return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallbackV2);
    }
    OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask in QuantAllReduce");
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2A5GenTaskUtils::Mc2GenTaskCallBack910A5);
}

IMPL_OP_CT(QuantAllReduce)
    .CalcOpParam(QuantAllReduceCalcOpParam)
    .GenerateTask(QuantAllReduceGenTask);

REGISTER_EXT_TASK_TYPE(QuantAllReduce, fe::ExtTaskType::kAicoreTask);

#endif

} // namespace ops
