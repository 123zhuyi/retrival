/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_circular_pad2d_backward.h"
#include "circular_pad_backward_common.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding,
                       const aclTensor* gradInput)
{
    auto selfDimnum = self->GetViewShape().GetDimNum();
    // self和gradInput的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, gradInput, return false);

    // 3 and 4 are dims, self只支持3维和4维
    OP_CHECK_MIN_DIM(self, 3, return false);
    OP_CHECK_MAX_DIM(self, 4, return false);

    // gradOutput, self, gradInput维度需要一致
    OP_CHECK(
        gradOutput->GetViewShape().GetDimNum() == selfDimnum && gradInput->GetViewShape().GetDimNum() == selfDimnum,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput, self, gradInput dim should be same."), return false);

    // padding长度为4
    OP_CHECK(padding->Size() == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 4, but got %zu.", padding->Size()),
             return false);

    // check the last 2 dim value of out. 0, 1, 2, 3 are indexes
    OP_CHECK(gradOutput->GetViewShape().GetDim(selfDimnum - 2) ==
                     self->GetViewShape().GetDim(selfDimnum - 2) + (*padding)[2] + (*padding)[3] &&
                 gradOutput->GetViewShape().GetDim(selfDimnum - 1) ==
                     self->GetViewShape().GetDim(selfDimnum - 1) + (*padding)[0] + (*padding)[1],
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "wrong gradOutput shape."), return false);
    return true;
}

static aclnnStatus commonPad2dBackward(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding,
                                       const string& mode, aclTensor* gradInput, uint64_t* workspaceSize,
                                       aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, padding, gradInput, CheckShape, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (gradOutput->IsEmpty() || self->IsEmpty() || gradInput->IsEmpty()) {
        *workspaceSize = 0;
        // 3 is dim number
        if (self->GetViewShape().GetDimNum() == 3) {
            // 1, 2 are indexes
            if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input should not be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 4 is dim number
        if (self->GetViewShape().GetDimNum() == 4) {
            // 1, 2 are indexes
            if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0 ||
                // 3 is index
                self->GetViewShape().GetDim(3) == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input should not be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用l0算子进行计算
    auto dim = self->GetViewShape().GetDimNum();
    ret = InputPreprocess(gradOutput, self, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    dim = self->GetViewShape().GetDimNum();
    auto paddingsTensor = GetPaddingTensor(dim, padding, uniqueExecutor.get());
    const aclTensor* pad2dbackwardResult = nullptr;
    pad2dbackwardResult = l0op::PadV3Grad(gradOutput, paddingsTensor, mode, true, true, uniqueExecutor.get());
    CHECK_RET(pad2dbackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参gradInput是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(pad2dbackwardResult, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCircularPad2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                       const aclIntArray* padding, aclTensor* gradInput,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnCircularPad2dBackward, DFX_IN(gradOutput, self, padding), DFX_OUT(gradInput));
    return commonPad2dBackward(gradOutput, self, padding, CIRCULAR_MODE, gradInput, workspaceSize, executor);
}

aclnnStatus aclnnCircularPad2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCircularPad2dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
