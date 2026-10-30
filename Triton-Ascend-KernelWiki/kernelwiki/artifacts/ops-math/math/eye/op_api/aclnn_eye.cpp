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
 * \file aclnn_eye.cpp
 * \brief
 */

#include "aclnn_eye.h"
#include "eye.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_FLOAT16, DataType::DT_INT8,
    DataType::DT_INT16, DataType::DT_UINT8, DataType::DT_INT64,   DataType::DT_BOOL};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_A2 = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_INT16,
    DataType::DT_UINT8, DataType::DT_INT64, DataType::DT_BOOL,    DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
        return DTYPE_SUPPORT_LIST;
    } else {
        return DTYPE_SUPPORT_LIST_A2;
    }
}

static const size_t OUT_DIM = 2;

static bool CheckNotNull(const aclTensor* out)
{
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* out)
{
    auto dtypeSupportList = GetDtypeSupportList();
    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);
    return true;
}

static bool CheckShape(const aclTensor* out, const int64_t n, const int64_t m)
{
    auto outShape = out->GetViewShape();
    auto outDim = outShape.GetDimNum();
    // out的dim应该为2
    if (outDim != OUT_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected out to be 2-D tensor, but the dimension of out is %zu.", outDim);
        return false;
    }

    if (n < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "n must be greater or equal to 0, got %ld", n);
        return false;
    }

    if (m < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "m must be greater or equal to 0, got %ld", m);
        return false;
    }

    if (outShape.GetDim(0) != n || outShape.GetDim(1) != m) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected out shape to be (%ld, %ld), but got %s.", n, m,
            op::ToString(outShape).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* out, const int64_t n, const int64_t m)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(out, n, m), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEyeGetWorkspaceSize(
    int64_t n, int64_t m, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnEye, DFX_IN(n, m), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(out, n, m);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (n == 0 || m == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用l0算子Eye进行计算
    auto eyeResult = l0op::Eye(out, n, m, uniqueExecutor.get());
    CHECK_RET(eyeResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(eyeResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEye(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnEye);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
