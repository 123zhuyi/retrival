/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/**
 * @file aclnn_xlog1py.cpp
 * @brief ACLNN L2 API 实现 - xlog1py 算子
 *
 * 标准两段式流程:
 * 1. GetWorkspaceSize: 参数检查 → Contiguous → L0算子 → ViewCopy → 返回workspace
 * 2. Execute: 执行计算
 */

#include "aclnn_xlog1py.h"
#include "xlog1py.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool HasEmptyTensor(const aclTensor* x, const aclTensor* y)
{
    return x->IsEmpty() || y->IsEmpty();
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(z, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    OP_CHECK_DTYPE_NOT_MATCH(x, y->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(z, x->GetDataType(), return false);

    OP_CHECK(IsDtypeSupported(x->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Dtype not supported: dtype=%d. Supported: FLOAT, FLOAT16, BF16.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    auto fmtX  = x->GetStorageFormat();
    auto fmtY  = y->GetStorageFormat();
    auto fmtZ = z->GetStorageFormat();

    OP_CHECK(!(IsPrivateFormat(fmtX) || IsPrivateFormat(fmtY) || IsPrivateFormat(fmtZ)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: x=%d, y=%d, z=%d",
                     static_cast<int>(fmtX), static_cast<int>(fmtY), static_cast<int>(fmtZ)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(y, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(z,   ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    CHECK_COND(CheckNotNull(x, y, z), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(x, y, z), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: x_dtype=%d, y_dtype=%d, z_dtype=%d",
               static_cast<int>(x->GetDataType()), static_cast<int>(y->GetDataType()),
               static_cast<int>(z->GetDataType()));
    CHECK_COND(CheckFormat(x, y, z), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: x_fmt=%d, y_fmt=%d, z_fmt=%d",
               static_cast<int>(x->GetStorageFormat()), static_cast<int>(y->GetStorageFormat()),
               static_cast<int>(z->GetStorageFormat()));
    CHECK_COND(CheckShape(x, y, z), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: x_dim=%zu, y_dim=%zu, z_dim=%zu",
               x->GetViewShape().GetDimNum(), y->GetViewShape().GetDimNum(),
               z->GetViewShape().GetDimNum());
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnXlog1pyGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* y,
    const aclTensor* z,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnXlog1py, DFX_IN(x, y), DFX_OUT(z));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, y, z);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x, y)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto yContiguous = l0op::Contiguous(y, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Xlog1py(xContiguous, yContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, z, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnXlog1py(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnXlog1py);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
