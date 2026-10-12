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
 * \file matmul_all_reduce_add_rms_norm_gen_task.cpp
 * \brief
 */
#include <vector>

#include "op_mc2.h"
#include "platform/platform_info.h"

#ifdef BUILD_OPEN_PROJECT
#include "mc2_gen_task_ops_utils.h"
#include "matmul_all_reduce_gen_task_ops_utils.h"
#include "graph/arg_desc_info.h"
#include "graph/kernel_launch_info.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#else
#include "matmul_all_reduce_gen_task_utils.h"
#include "mc2_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#endif

namespace ops {

#ifdef BUILD_OPEN_PROJECT
ge::Status MatmulAllReduceAddRmsNormCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MatmulAllReduceAddRmsNormGenTaskFunc(const gert::ExeResGenerationContext *context,
                                                std::vector<std::vector<uint8_t>> &tasks)
{
    return MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(context, tasks);
}

// new ver
IMPL_OP(MatmulAllReduceAddRmsNorm)
    .CalcOpParam(MatmulAllReduceAddRmsNormCalcParamFunc)
    .GenerateTask(MatmulAllReduceAddRmsNormGenTaskFunc);
#else // mc2 gen task utils
ge::Status MatmulAllReduceAddRmsNormCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MatmulAllReduceAddRmsNormGenTaskFunc(const gert::ExeResGenerationContext *context,
                                                std::vector<std::vector<uint8_t>> &tasks)
{
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks,
                                                MatmulAllReduceGenTaskUtils::MatmulAllReduceGenTaskCallback);
}

IMPL_OP_CT(MatmulAllReduceAddRmsNorm)
    .CalcOpParam(MatmulAllReduceAddRmsNormCalcParamFunc)
    .GenerateTask(MatmulAllReduceAddRmsNormGenTaskFunc);
#endif
} // namespace ops
