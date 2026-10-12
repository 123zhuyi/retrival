/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ceil.h"
#include "ceil.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // self和out数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    // 检查self的数据类型是否在支持列表内
    auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    return true;
}

static inline aclnnStatus CheckParamsCeil(const aclTensor* self, const aclTensor* out)
{
    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查shape是否满足约束
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecCeilGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_NOT_NULL(self, out);
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsCeil(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子Ceil进行计算
    auto ceilResult = l0op::Ceil(selfContiguous, uniqueExecutor.get());
    CHECK_RET(ceilResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(ceilResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCeilGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCeil, DFX_IN(self), DFX_OUT(out));
    return ExecCeilGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceCeilGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceCeil, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    return ExecCeilGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnCeil(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCeil);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceCeil(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceCeil);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif