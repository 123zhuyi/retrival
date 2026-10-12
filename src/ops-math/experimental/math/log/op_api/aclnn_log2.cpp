/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_log2.h"
#include "log.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
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
#include "aclnn_kernels/common/op_error_check.h"
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

// 根据API定义，需要列出1980所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,     op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,      op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16,  op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16,   op::DataType::DT_BF16,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 获取芯片类型,判断是1971还是1980
    bool is910bSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E);
    const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST =
        is910bSocVersion ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;
    const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST =
        is910bSocVersion ? ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST : ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST;

    // 检查self的数据类型是否在log算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUTPUT_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckInplaceDtypeValid(const aclTensor* selfRef)
{
    auto inplaceSupportList =
        GetDtypeSupportListV2(ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST, ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST);
    // 检查selfRef的数据类型是否在inplace log2算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

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
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParams(const aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace log2算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnLog2Common(
    const aclTensor* self, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
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

    const float LOG2_BASE = 2.0f;
    const float LOG2_SCALE = 1.0f;
    const float LOG2_SHIFT = 0.0f;
    auto logOut = l0op::Log(selfContiguous, LOG2_BASE, LOG2_SCALE, LOG2_SHIFT, uniqueExecutor.get());
    CHECK_RET(logOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(logOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLog2GetWorkspaceSize(
    const aclTensor* self, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLog2, DFX_IN(self), DFX_OUT(out));

    return aclnnLog2Common(self, out, workspaceSize, executor);
}

aclnnStatus aclnnLog2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLog2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLog2GetWorkspaceSize(
    const aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceLog2, DFX_IN(selfRef), DFX_OUT(selfRef));

    auto out = const_cast<aclTensor*>(selfRef);
    auto ret = CheckInplaceParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return aclnnLog2Common(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceLog2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceLog2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
