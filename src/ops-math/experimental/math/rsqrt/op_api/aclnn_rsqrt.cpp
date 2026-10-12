/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_rsqrt.cpp
 * \brief
 */

#include "aclnn_rsqrt.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "rsqrt.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"
#include "op_api/level2_base_caculation.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_INT16,
    op::DataType::DT_INT32, op::DataType::DT_BOOL,    op::DataType::DT_BF16, op::DataType::DT_UINT8};

static bool CheckInplaceDtypeValid(aclTensor* selfRef)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

inline static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    if (self->GetDataType() != out->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype must be same as Out dtype.");
        return false;
    }
    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && self->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s is unsupported by the current SOC version [%s].",
                op::ToString(self->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParamsRsqrt(const aclTensor* self, const aclTensor* out)
{
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParamsRsqrt(aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecRsqrtGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParamsRsqrt(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto rsqrtOpOut = l0op::Rsqrt(selfContiguous, uniqueExecutor.get());
    CHECK_RET(rsqrtOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(rsqrtOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRsqrtGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                       aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRsqrt, DFX_IN(self), DFX_OUT(out));
    return ExecRsqrtGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceRsqrtGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceRsqrt, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto ret = CheckInplaceParamsRsqrt(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto out = const_cast<aclTensor*>(selfRef);
    return ExecRsqrtGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnRsqrt(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRsqrt);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceRsqrt(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceRsqrt);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
