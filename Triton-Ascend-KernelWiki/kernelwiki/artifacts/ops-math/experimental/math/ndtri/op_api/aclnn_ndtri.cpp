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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/**
 * @file aclnn_ndtri.cpp
 * @brief ACLNN L2 API 实现 - Ndtri
 */

#include "aclnn_ndtri.h"
#include "ndtri.h"
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

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto dtype = self->GetDataType();
    OP_CHECK(
        CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Ndtri unsupported dtype: %d. Supported: FLOAT, FLOAT16, BF16.",
            static_cast<int>(dtype)),
        return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, dtype, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed");
    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnNdtriGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNdtri, DFX_IN(self), DFX_OUT(out));

    // 0. 基础空指针检查（不创建 Executor）
    if (workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 1. 参数校验
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 2. 空 Tensor short-circuit
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 3. Contiguous
    auto selfContig = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 4. L0 API
    auto l0Out = l0op::Ndtri(selfContig, uniqueExecutor.get());
    CHECK_RET(l0Out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 5. ViewCopy 到用户 out
    auto viewOut = l0op::ViewCopy(l0Out, out, uniqueExecutor.get());
    CHECK_RET(viewOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnNdtri(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNdtri);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
