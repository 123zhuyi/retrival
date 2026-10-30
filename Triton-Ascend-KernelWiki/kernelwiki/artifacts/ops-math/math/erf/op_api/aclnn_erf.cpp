/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_erf.h"
#include "erf.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/level2_base.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910 = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BOOL, op::DataType::DT_INT64};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910B = {DataType::DT_DOUBLE,  DataType::DT_FLOAT,
                                                                        DataType::DT_FLOAT16, DataType::DT_BF16,
                                                                        DataType::DT_BOOL,    op::DataType::DT_INT64};

static const std::initializer_list<DataType> ASCEND910_DTYPE_SELFREF_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SELFREF_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 获取芯片类型和支持列表
    const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = GetDtypeSupportListV2(DTYPE_SUPPORT_LIST_910B, DTYPE_SUPPORT_LIST_910);

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    // 当self的dtype为bool或int64时，输出dtype（float）不能cast为out的dtype
    if (self->GetDataType() == op::DataType::DT_BOOL || self->GetDataType() == op::DataType::DT_INT64) {
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(op::DataType::DT_FLOAT, out->GetDataType(), return false);
    }

    // 当self的dtype不为bool或int64时，输出dtype（输入dtype）不能cast为out的dtype
    if (self->GetDataType() != op::DataType::DT_BOOL && self->GetDataType() != op::DataType::DT_INT64) {
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
    }

    return true;
}

static bool CheckInplaceDtypeValid(const aclTensor* selfRef)
{
    auto inplaceSupportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SELFREF_LIST, ASCEND910_DTYPE_SELFREF_LIST);
    // 检查selfRef的数据类型是否在inplace erf算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static aclnnStatus CheckParamsErf(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParamsErf(aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace erf算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsErf(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castSelf = selfContiguous;
    if (self->GetDataType() == op::DataType::DT_BOOL || self->GetDataType() == op::DataType::DT_INT64) {
        castSelf = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    }

    // 调用l0算子Erf进行计算
    auto erfResult = l0op::Erf(castSelf, uniqueExecutor.get());
    CHECK_RET(erfResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将结果cast成out的dtype类型
    auto castErfResult = l0op::Cast(erfResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castErfResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(castErfResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnErfGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnErf, DFX_IN(self), DFX_OUT(out));
    return GetWorkspaceSizeCommon(self, out, workspaceSize, executor);
}

aclnnStatus aclnnErf(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnErf);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceErfGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceErf, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto ret = CheckInplaceParamsErf(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return GetWorkspaceSizeCommon(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceErf(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceErf);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}