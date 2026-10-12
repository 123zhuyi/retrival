/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_reflection_pad1d_backward.h"
#include "../../../pad_v3_grad_replicate/op_host/op_api/pad1d_backward_common.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const string REFLECTION_MODE = "reflect";

aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                         const aclIntArray* padding, aclTensor* gradInput,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnReflectionPad1dBackward, DFX_IN(gradOutput, self, padding), DFX_OUT(gradInput));
    return Pad1dBackwardCompute(gradOutput, self, padding, gradInput, workspaceSize, executor,
                                Pad1dBackwardMode::REFLECT, REFLECTION_MODE, "[PadV4Grad]");
}

aclnnStatus aclnnReflectionPad1dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReflectionPad1dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
