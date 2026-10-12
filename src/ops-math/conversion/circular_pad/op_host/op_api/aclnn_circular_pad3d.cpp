/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_circular_pad3d.h"
#include "circular_pad_common.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCircularPad3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* padding, aclTensor* out,
                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnCircularPad3d, DFX_IN(self, padding), DFX_OUT(out));
    // 3d: minDim=4, maxDim=5, paddingSize=6
    return CircularPadGetWorkspaceSizeImpl<4, 5, 6>(self, padding, out, workspaceSize, executor);
}

aclnnStatus aclnnCircularPad3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCircularPad3d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
