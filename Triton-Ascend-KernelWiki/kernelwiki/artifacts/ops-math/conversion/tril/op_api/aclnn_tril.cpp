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
 * \file aclnn_tril.cpp
 * \brief
 */
#include "tril.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_tril.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_UINT8, op::DataType::DT_INT8,    op::DataType::DT_INT16,  op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_UINT16,  op::DataType::DT_UINT32, op::DataType::DT_UINT64};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_UINT8, op::DataType::DT_INT8,    op::DataType::DT_INT16,  op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16,    op::DataType::DT_UINT16, op::DataType::DT_UINT32,
    op::DataType::DT_UINT64};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_UINT8,  op::DataType::DT_INT8,      op::DataType::DT_INT16,    op::DataType::DT_INT32,
    op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,    op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,   op::DataType::DT_BF16,      op::DataType::DT_UINT16,   op::DataType::DT_UINT32,
    op::DataType::DT_UINT64, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX32};

static inline const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return DTYPE_SUPPORT_LIST_910B;
    } else if (IsRegBase()) {
        return DTYPE_SUPPORT_LIST_REGBASE;
    }
    return DTYPE_SUPPORT_LIST_910;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    if (self->GetViewShape().GetDimNum() <= 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of self should be last 2 dims.");
        return false;
    }

    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);

    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    // self和out的format必须一致
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self format is different with out format, self [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self的数据类型能否转换为输出数据类型
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTrilGetWorkspaceSize(
    const aclTensor* self, int64_t diagonal, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnTril, DFX_IN(self, diagonal), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子Tril行计算
    auto trilResult = l0op::Tril(selfContiguous, diagonal, uniqueExecutor.get());
    CHECK_RET(trilResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(trilResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceTrilGetWorkspaceSize(
    const aclTensor* selfRef, int64_t diagonal, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnTrilGetWorkspaceSize(selfRef, diagonal, out, workspaceSize, executor);
}

aclnnStatus aclnnTril(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTril);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceTril(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceTril);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
