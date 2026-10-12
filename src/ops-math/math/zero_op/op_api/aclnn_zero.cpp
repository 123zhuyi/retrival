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
 * \file aclnn_zero.cpp
 * \brief
 */

#include "aclnn_zero.h"
#include "aclnn_kernels/contiguous.h"
#include "zero_op.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,    op::DataType::DT_INT32,  op::DataType::DT_INT64,      op::DataType::DT_UINT8,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BOOL,       op::DataType::DT_DOUBLE,
    op::DataType::DT_INT16,   op::DataType::DT_UINT16, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_950_LIST = {
    op::DataType::DT_INT8,     op::DataType::DT_INT32,       op::DataType::DT_INT64,
    op::DataType::DT_UINT8,    op::DataType::DT_FLOAT16,     op::DataType::DT_FLOAT,
    op::DataType::DT_BOOL,     op::DataType::DT_DOUBLE,      op::DataType::DT_INT16,
    op::DataType::DT_UINT16,   op::DataType::DT_COMPLEX128,  op::DataType::DT_COMPLEX64,
    op::DataType::DT_BF16,     op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT4_E2M1};

static bool CheckNotNull(const aclTensor* self)
{
    OP_CHECK_NULL(self, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self)
{
    if (IsRegBase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_950_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceZeroGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceZero, DFX_IN(selfRef), DFX_OUT(selfRef));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // ZerosLike算子的空tensor在kernel中支持
    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用ZerosLike算子kernel
    auto zeroOut = l0op::ZerosLike(selfContiguous, uniqueExecutor.get());
    CHECK_RET(zeroOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewcopyResult = l0op::ViewCopy(zeroOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceZero(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceZero);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
