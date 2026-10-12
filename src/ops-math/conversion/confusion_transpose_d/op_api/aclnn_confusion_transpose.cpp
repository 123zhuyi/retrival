/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_confusion_transpose.h"
#include "confusion_transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,    op::DataType::DT_INT16,  op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8,   op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclIntArray* perm, const aclIntArray* shape, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_COND((x != nullptr), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull x failed!");
    CHECK_COND((perm != nullptr), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull perm failed!");
    CHECK_COND((shape != nullptr), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull shape failed!");
    CHECK_COND((out != nullptr), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull out failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus aclnnConfusionTransposeGetWorkspaceSize(
    const aclTensor* x, const aclIntArray* perm, const aclIntArray* shape, bool transpose_first, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnConfusionTranspose, DFX_IN(x, perm, shape, transpose_first), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, perm, shape, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor直接返回
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 非连续转连续
    if (x->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        x = l0op::Contiguous(x, uniqueExecutor.get());
    }

    const aclTensor* result = l0op::ConfusionTransposeD(x, perm, shape, transpose_first, out, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 转移uniqueExecutor内部管理的executor所有权到输出参数executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConfusionTranspose(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConfusionTranspose);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif