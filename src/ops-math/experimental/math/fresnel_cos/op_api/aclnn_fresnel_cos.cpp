/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Disclaimer: This file is generated with the assistance of an AI tool.
 * Please review carefully before use.
 */

#include "aclnn_fresnel_cos.h"
#include "fresnel_cos.h"
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
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static bool IsDtypeSupported(DataType dtype) { return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST); }

static bool CheckNotNull(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return false);
    OP_CHECK(
        IsDtypeSupported(x->GetDataType()),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "FresnelCos dtype not supported: %d. Only FLOAT, FLOAT16, BFLOAT16.",
            static_cast<int>(x->GetDataType())),
        return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK(
        x->GetViewShape() == out->GetViewShape(),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FresnelCos: x and out shape must match."), return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(x, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(
        CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed: x_dtype=%d, out_dtype=%d",
        static_cast<int>(x->GetDataType()), static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFresnelCosGetWorkspaceSize(
    const aclTensor* x, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFresnelCos, DFX_IN(x), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::FresnelCos(xContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFresnelCos(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFresnelCos);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
