/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_trunc.h"
#include "aclnn_kernels/contiguous.h"
#include "trunc.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"
#include "op_api/aclnn_check.h"


using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ARCH3510_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> GetDtypeSupportListSelfTri(
    const std::initializer_list<op::DataType>& l1, const std::initializer_list<op::DataType>& l2,
    const std::initializer_list<op::DataType>& l3)
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        if (IsRegBase()) {
            return l3;
        }
        return l1;
    } else {
        return l2;
    }
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    const auto& supportList = GetDtypeSupportListSelfTri(
        ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST, ARCH3510_DTYPE_DTYPE_SUPPORT_LIST);
    // 检查self的数据类型是否在trunc算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    // 检查other的数据类型是否在trunc算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    // 检查self和other的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static aclnnStatus CheckParamsTrunc(const aclTensor *self, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. ND 算子不检查格式
    // 4. 检查self和out的shape是否一致
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecTruncGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsTrunc(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // trunc算子的空tensor在kernel中不支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        OP_LOGD("empty input tensor");
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Trunc算子Kernel
    auto truncOpOut = l0op::Trunc(selfContiguous, uniqueExecutor.get());
    CHECK_RET(truncOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(truncOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTruncGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    
    L2_DFX_PHASE_1(aclnnTrunc, DFX_IN(self), DFX_OUT(out));
    return ExecTruncGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceTruncGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    
    L2_DFX_PHASE_1(aclnnInplaceTrunc, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    return ExecTruncGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnTrunc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnTrunc);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceTrunc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceTrunc);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif