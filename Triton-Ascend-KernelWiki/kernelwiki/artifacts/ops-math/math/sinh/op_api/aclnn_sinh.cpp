/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sinh.h"
#include "sinh.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"


using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8,      op::DataType::DT_INT8,   op::DataType::DT_INT16,
    op::DataType::DT_INT32,      op::DataType::DT_INT64,      op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8,      op::DataType::DT_INT8,   op::DataType::DT_INT16,
    op::DataType::DT_INT32,      op::DataType::DT_INT64,      op::DataType::DT_BOOL,   op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_OUT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_OUT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static bool CheckInplaceDtypeValid(aclTensor *selfRef) {
    auto inplaceSupportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_OUT_LIST, ASCEND910_DTYPE_OUT_LIST);
    // 检查selfRef的数据类型是否在inplace sinh算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    // 检查self的数据类型是否在sinh算子的支持列表内
    auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
    auto outSupportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_OUT_LIST, ASCEND910_DTYPE_OUT_LIST);
    CHECK_RET(CheckDtypeValid1In1Out(self, out, supportList, outSupportList), ACLNN_ERR_PARAM_INVALID);

    // 3. 算子检查格式
    if (self->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGW("Only support ND format for sinh/inplaceSinh operator.");
    }

    // 4. 检查self和out的shape是否一致
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParams(aclTensor *selfRef) {
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);
    // 检查selfRef的数据类型是否在inplace sinh算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecSinhGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 支持空tensor
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将self转为可支持的类型，或者out类型
    auto outSupportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_OUT_LIST, ASCEND910_DTYPE_OUT_LIST);
    auto castDtype = selfContiguous->GetDataType();
    if (!CheckType(castDtype, outSupportList)) {
        castDtype = out->GetDataType();
    }
    auto selfCast = l0op::Cast(selfContiguous, castDtype, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Sinh算子Kernel
    auto sinhOpOut = l0op::Sinh(selfCast, uniqueExecutor.get());
    CHECK_RET(sinhOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果转换成输出的数据类型
    auto castOut = l0op::Cast(sinhOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSinhGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSinh, DFX_IN(self), DFX_OUT(out));
    return ExecSinhGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceSinhGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnInplaceSinh, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto ret = CheckInplaceParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecSinhGetWorkspaceSize(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnSinh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnSinh);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceSinh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceSinh);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif