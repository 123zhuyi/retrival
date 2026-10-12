/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_erfc.h"
#include "erfc.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"

using namespace op;

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL,
    op::DataType::DT_INT64};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
    op::DataType::DT_BOOL,   op::DataType::DT_INT64, op::DataType::DT_BF16};

static const std::initializer_list<DataType> ASCEND910_DTYPE_SELFREF_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SELFREF_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在支持列表内
    auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

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
    // 检查selfRef的数据类型是否在inplace erfc算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static aclnnStatus CheckParamsErfc(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParamsErfc(const aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace erfc算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnErfcGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnErfc, DFX_IN(self), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsErfc(self, out);
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
        CHECK_RET(castSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0算子Erfc进行计算
    auto erfcResult = l0op::Erfc(castSelf, uniqueExecutor.get());
    CHECK_RET(erfcResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将结果cast成out的dtype类型
    auto castErfcResult = l0op::Cast(erfcResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castErfcResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(castErfcResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceErfcGetWorkspaceSize(const aclTensor* selfRef, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    auto ret = CheckInplaceParamsErfc(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return aclnnErfcGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnErfc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnErfc);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceErfc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceErfc);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
