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
 * \file aclnn_trace.cpp
 * \brief
 */

#include "aclnn_trace.h"
#include "trace.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_INT32,   op::DataType::DT_INT64,  op::DataType::DT_INT16,
    op::DataType::DT_INT8,       op::DataType::DT_UINT8,   op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_INT32,   op::DataType::DT_INT64,  op::DataType::DT_INT16,
    op::DataType::DT_INT8,       op::DataType::DT_UINT8,   op::DataType::DT_BOOL,   op::DataType::DT_BF16};

static const size_t DIM_SUPPORT_ONLY = 2;

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportList();

    // 检查self的数据类型是否在trace算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    auto outDtype = (IsIntegralType(self->GetDataType(), true)) ? op::DataType::DT_INT64 : self->GetDataType();
    // 检查out的数据类型
    OP_CHECK_DTYPE_NOT_MATCH(out, outDtype, return false);

    return true;
}

static bool CheckParamValid(const aclTensor* self, const aclTensor* out)
{
    // 检查参数dim是否合法
    auto dim = self->GetViewShape().GetDimNum();
    if (dim != DIM_SUPPORT_ONLY) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "trace: expected a matrix, but got tensor with dim %zu", dim);
        return false;
    }

    dim = out->GetViewShape().GetDimNum();
    if (dim != 0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "expected 0D output tensor, but got %s tensor.",
            op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据dim是否合法
    CHECK_RET(CheckParamValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTraceGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnTrace, DFX_IN(self), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const aclTensor* traceOpOut;
    // 空tensor
    if (self->IsEmpty()) {
        const aclScalar* valueScalar = (uniqueExecutor.get())->AllocScalar(0);
        CHECK_RET(valueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        traceOpOut = (uniqueExecutor.get())->ConvertToTensor(valueScalar, out->GetDataType());
    } else {
        // 固定写法，将输入self转换成连续的tensor
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto inputdtype = selfContiguous->GetDataType();
        bool isNeedCast = (inputdtype == DataType::DT_FLOAT16);
        auto selfCast =
            isNeedCast ? l0op::Cast(selfContiguous, DataType::DT_FLOAT, uniqueExecutor.get()) : selfContiguous;
        CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用trace进行计算
        auto traceOpOutCast = l0op::Trace(selfCast, uniqueExecutor.get());

        traceOpOut = isNeedCast ? l0op::Cast(traceOpOutCast, out->GetDataType(), uniqueExecutor.get()) : traceOpOutCast;
    }
    CHECK_RET(traceOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(traceOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTrace(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTrace);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif