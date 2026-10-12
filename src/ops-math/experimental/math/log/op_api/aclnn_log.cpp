/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_log.h"
#include "log.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* Log 算子的完整计算流程如下:
 *           self
 *            |
 * Contiguous(workspace_0)
 *            |
 *     Cast(workspace_1)
 *            |
 *      Log(workspace_4)
 *            |
 *     Cast(workspace_5)
 *            |
 *        ViewCopy
 *            |
 *          result
 */

constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL,    op::DataType::DT_INT64,  op::DataType::DT_INT32,
    op::DataType::DT_INT16,      op::DataType::DT_INT8,    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_FLOAT16,    op::DataType::DT_BF16, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL, op::DataType::DT_INT64,
    op::DataType::DT_INT32,     op::DataType::DT_INT16,      op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16,   op::DataType::DT_BF16,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static bool CheckPromoteType(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    auto inputSupportList =
        GetDtypeSupportListV2(ASCEND910B_INPUT_DTYPE_SUPPORT_LIST, ASCEND910_INPUT_DTYPE_SUPPORT_LIST);
    auto outputSupportList =
        GetDtypeSupportListV2(ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST, ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST);
    CHECK_RET(CheckDtypeValid1In1Out(self, out, inputSupportList, outputSupportList), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLog, DFX_IN(self, out), DFX_OUT());

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // log算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const float LOG_BASE = -1.0f;
    const float LOG_SCALE = 1.0f;
    const float LOG_SHIFT = 0.0f;
    auto logOut = l0op::Log(selfContiguous, LOG_BASE, LOG_SCALE, LOG_SHIFT, uniqueExecutor.get());
    CHECK_RET(logOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(logOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckInplace(const aclTensor* selfRef)
{
    auto outputSupportList =
        GetDtypeSupportListV2(ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST, ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST);
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, outputSupportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(selfRef, MAX_DIM_LEN, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceLogGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckInplace(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return aclnnLogGetWorkspaceSize(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnLog(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLog);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLog(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceLog);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
