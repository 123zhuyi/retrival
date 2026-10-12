/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_ones.h"
#include "ones_like.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32,   op::DataType::DT_INT64,
    op::DataType::DT_UINT8, op::DataType::DT_BOOL,    op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16, op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,  op::DataType::DT_INT64,   op::DataType::DT_UINT8, op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self)
{
    OP_CHECK_NULL(self, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self)
{
    // 检查self的数据类型是否在支持列表内
    bool isAscend910BSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
         IsRegBase());
    const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST =
        isAscend910BSocVersion ? ASCEND910B_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_SUPPORT_LIST;

    OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckMaxDimension(const aclTensor* self)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查self的数据类型是否合法
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查最大维度是否超过阈值
    CHECK_RET(CheckMaxDimension(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceOneGetWorkspaceSize(const aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceOne, DFX_IN(selfRef), DFX_OUT(selfRef));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor在kernel中支持
    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用OnesLike算子kernel
    auto onesOut = l0op::OnesLike(selfContiguous, uniqueExecutor.get());
    CHECK_RET(onesOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(onesOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceOne(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceOne);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
