/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_logical_and.h"
#include "logical_and.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* LogicalAnd 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             LogicalAnd(workspace_4)
 *                    |
 *               Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

constexpr size_t MAX_DIM_LEN = 8;

// 所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16,  op::DataType::DT_INT8,    op::DataType::DT_UINT8,     op::DataType::DT_INT16,
    op::DataType::DT_INT64, op::DataType::DT_BOOL,    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

inline static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other)
{
    // 检查self的数据类型是否在logical_and算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    // 检查other的数据类型是否在logical_and算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, DTYPE_SUPPORT_LIST, return false);
    return true;
}

inline static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(other, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查双输入是否能broadcast
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CalculateResult(
    const aclTensor* self, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    // 参数检查
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || other->IsEmpty()) {
        return ACLNN_SUCCESS;
    }

    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成DT_BOOL类型
    auto selfCasted = (selfContiguous->GetDataType() == op::DataType::DT_BOOL) ?
                          selfContiguous :
                          l0op::Cast(selfContiguous, op::DataType::DT_BOOL, executor);
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, executor);
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成DT_BOOL类型
    auto otherCasted = (otherContiguous->GetDataType() == op::DataType::DT_BOOL) ?
                           otherContiguous :
                           l0op::Cast(otherContiguous, op::DataType::DT_BOOL, executor);
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行LogicalAnd计算
    auto logical_andOpOut = l0op::LogicalAnd(selfCasted, otherCasted, executor);
    CHECK_RET(logical_andOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果转换成输出out的数据类型
    auto castOut = (out->GetDataType() == op::DataType::DT_BOOL) ?
                       logical_andOpOut :
                       l0op::Cast(logical_andOpOut, out->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogicalAndGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLogicalAnd, DFX_IN(self, other), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CalculateResult(self, other, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceLogicalAndGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceLogicalAnd, DFX_IN(selfRef, other), DFX_OUT(selfRef));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CalculateResult(selfRef, other, selfRef, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogicalAnd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLogicalAnd);
    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLogicalAnd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceLogicalAnd);
    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
