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
 * @file aclnn_acos_grad_v2.cpp
 * @brief ACLNN L2 API 实现 - AcosGradV2 算子 (A2 / Ascend910B)
 *
 * z = -dy / sqrt(1 - y^2)
 */

#include "aclnn_acos_grad_v2.h"
#include "acos_grad_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> ACOS_GRAD_V2_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsDtypeSupported(DataType dtype)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    // A2 (Atlas A2 训练/推理系列产品) -> DAV_2201
    if (npuArch == NpuArch::DAV_2201) {
        return CheckType(dtype, ACOS_GRAD_V2_DTYPE_SUPPORT_LIST);
    }
    return false;
}

static bool HasEmptyTensor(const aclTensor* y, const aclTensor* dy) { return y->IsEmpty() || dy->IsEmpty(); }

static bool CheckNotNull(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(dy, return false);
    OP_CHECK_NULL(z, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    OP_CHECK_DTYPE_NOT_MATCH(dy, y->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(z, y->GetDataType(), return false);

    if (!IsDtypeSupported(y->GetDataType())) {
        auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AcosGradV2: Dtype not supported: dtype=%d, npuArch=%d. "
                "Supported: FLOAT16, FLOAT32, BF16.",
                static_cast<int>(y->GetDataType()), static_cast<int>(npuArch));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    auto fmtY = y->GetStorageFormat();
    auto fmtDy = dy->GetStorageFormat();
    auto fmtZ = z->GetStorageFormat();

    if (IsPrivateFormat(fmtY) || IsPrivateFormat(fmtDy) || IsPrivateFormat(fmtZ)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGradV2: Private format not supported: y=%d, dy=%d, z=%d",
                static_cast<int>(fmtY), static_cast<int>(fmtDy), static_cast<int>(fmtZ));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    OP_CHECK_MAX_DIM(y, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(dy, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(z, ACLNN_MAX_SHAPE_RANK, return false);

    auto yShape = y->GetViewShape();
    auto dyShape = dy->GetViewShape();
    auto zShape = z->GetViewShape();

    if (yShape != dyShape || yShape != zShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGradV2: Shape mismatch: y=%s, dy=%s, z=%s",
                op::ToString(yShape).GetString(), op::ToString(dyShape).GetString(), op::ToString(zShape).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    if (!CheckNotNull(y, dy, z)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "AcosGradV2: CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(y, dy, z)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGradV2: CheckDtypeValid failed: y_dtype=%d, dy_dtype=%d, z_dtype=%d",
                static_cast<int>(y->GetDataType()), static_cast<int>(dy->GetDataType()),
                static_cast<int>(z->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(y, dy, z)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGradV2: CheckFormat failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(y, dy, z)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGradV2: CheckShape failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAcosGradV2GetWorkspaceSize(const aclTensor* y, const aclTensor* dy, const aclTensor* z,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAcosGradV2, DFX_IN(y, dy), DFX_OUT(z));

    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(y, dy, z);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(y, dy)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto yContiguous = l0op::Contiguous(y, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dyContiguous = l0op::Contiguous(dy, uniqueExecutor.get());
    CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::AcosGradV2(yContiguous, dyContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, z, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAcosGradV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAcosGradV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
