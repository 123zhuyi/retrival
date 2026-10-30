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
 * \file allto_allv_grouped_mat_mul_gen_task_training.cpp
 * \brief
 */
#include <vector>

#ifdef BUILD_OPEN_PROJECT
#include "mc2_gen_task_ops_utils.h"
#include "mc2_moe_gen_task_ops_utils.h"
#include "mc2_gen_task_ops_utils_arch35.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#else
#include "mc2_gen_task_training.h"
#include "mc2_gen_task_utils.h"
#include "mc2_a5_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#endif

namespace ops {
#ifdef BUILD_OPEN_PROJECT
ge::Status AlltoAllvGroupedMatMulCalcParamFunc(gert::ExeResGenerationContext *context)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask CalcParamFunc");
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    OPS_LOG_D(context->GetNodeName(), "Do A3 aicpu GenTask CalcParamFunc");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status AlltoAllvGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask GenTaskFunc");
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    }
    OPS_LOG_D(context->GetNodeName(), "Do A3 aicpu GenTask GenTaskFunc");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
}

// new ver
IMPL_OP(AlltoAllvGroupedMatMul)
    .CalcOpParam(AlltoAllvGroupedMatMulCalcParamFunc)
    .GenerateTask(AlltoAllvGroupedMatMulGenTaskFunc);
#else // mc2 gen task utils
ge::Status AlltoAllvGroupedMatMulCalcParamFunc(gert::ExeResGenerationContext *context)
{
    if (Mc2A5GenTaskUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status AlltoAllvGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    if (Mc2A5GenTaskUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2A5GenTaskUtils::Mc2GenTaskCallBack910A5);
    }
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskTraining::Mc2TrainingGenTaskCallback);
}

IMPL_OP_CT(AlltoAllvGroupedMatMul)
    .CalcOpParam(AlltoAllvGroupedMatMulCalcParamFunc)
    .GenerateTask(AlltoAllvGroupedMatMulGenTaskFunc);
#endif
} // namespace ops

