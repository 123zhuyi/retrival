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

#include "aclnn_log_space.h"
#include "log_space.h"
#include <climits>
#include <limits>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

static const std::initializer_list<op::DataType> RESULT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static const std::initializer_list<op::DataType> SCALAR_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_BF16};

static bool CheckNotNull(
    const aclScalar* start, const aclScalar* end, const aclTensor* result, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_NULL(start, return false);
    OP_CHECK_NULL(end, return false);
    OP_CHECK_NULL(result, return false);
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize or executor is nullptr.");
        return false;
    }
    return true;
}

static bool CheckParamRange(
    const aclScalar* start, const aclScalar* end, int64_t steps, double base, const aclTensor* result)
{
    OP_CHECK(steps >= 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "steps must be >= 0, got %ld", steps), return false);
    OP_CHECK(
        steps <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "steps must be <= UINT32_MAX (%u), got %ld", std::numeric_limits<uint32_t>::max(),
            steps),
        return false);
    OP_CHECK(base > 0.0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "base must be > 0, got %f", base), return false);
    OP_CHECK(
        CheckType(start->GetDataType(), SCALAR_DTYPE_SUPPORT_LIST),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "start dtype %d not floating point", static_cast<int>(start->GetDataType())),
        return false);
    OP_CHECK(
        CheckType(end->GetDataType(), SCALAR_DTYPE_SUPPORT_LIST),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "end dtype %d not floating point", static_cast<int>(end->GetDataType())),
        return false);
    OP_CHECK(
        CheckType(result->GetDataType(), RESULT_DTYPE_SUPPORT_LIST),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "result dtype %d not in {FLOAT, FLOAT16, BFLOAT16}",
            static_cast<int>(result->GetDataType())),
        return false);

    // shape 校验：result 必须为 1D 且 shape == [steps]
    auto vshape = result->GetViewShape();
    OP_CHECK(
        vshape.GetDimNum() == 1, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "result must be 1D, got dim=%zu", vshape.GetDimNum()),
        return false);
    OP_CHECK(
        vshape.GetDim(0) == steps,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "result shape[0]=%ld not equal to steps=%ld", vshape.GetDim(0), steps),
        return false);
    return true;
}

extern "C" aclnnStatus aclnnLogSpaceGetWorkspaceSize(
    const aclScalar* start, const aclScalar* end, int64_t steps, double base, const aclTensor* result,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // Null pointer checks must be done BEFORE L2_DFX_PHASE_1, because the DFX
    // macro serializes IN/OUT params into cache hash and profiling, which would
    // dereference null start/end/result and crash with ACLNN_ERR_INNER_NULLPTR(561103).
    if (!CheckNotNull(start, end, result, workspaceSize, executor)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    L2_DFX_PHASE_1(aclnnLogSpace, DFX_IN(start, end, steps, base), DFX_OUT(result));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (!CheckParamRange(start, end, steps, base, result)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // steps==0 短路：返回成功，不下发 Kernel
    if (steps == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const float startF = start->ToFloat();
    const float endF = end->ToFloat();
    const float baseF = static_cast<float>(base);

    const aclTensor* opResult = l0op::LogSpace(startF, endF, steps, baseF, result, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnLogSpace(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLogSpace);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
