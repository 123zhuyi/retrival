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
 * \file aclnn_logspace.cpp
 * \brief
 */
#include "aclnn_logspace.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "math/lin_space/op_host/op_api/linspace.h"
#include "math/pow/op_api/pow.h"

using namespace op;

static const std::initializer_list<DataType> LOGSPACE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

static bool CheckNotNull(const aclScalar *start, const aclScalar *end, const aclTensor* result){
    OP_CHECK_NULL(start, return false);
    OP_CHECK_NULL(end, return false);
    OP_CHECK_NULL(result,return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* result){
    DataType result_dtype = result->GetDataType();
    return std::find(LOGSPACE_DTYPE_SUPPORT_LIST.begin(),
                    LOGSPACE_DTYPE_SUPPORT_LIST.end(),
                    result_dtype) != LOGSPACE_DTYPE_SUPPORT_LIST.end();
}

static bool CheckStepsValid(int64_t steps){
    if (steps < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,"LogSpace requires non-negative steps, given steps is %ld", steps);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclScalar *start, const aclScalar *end, int64_t steps, const aclTensor *result){
    CHECK_RET(CheckNotNull(start, end, result),ACLNN_ERR_INNER_NULLPTR);
    //检查数据类型支持
    CHECK_RET(CheckDtypeValid(result), ACLNN_ERR_PARAM_INVALID);
    //检查steps有效性
    CHECK_RET(CheckStepsValid(steps), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* ScalarToTensor(const aclScalar *other, const op::DataType dataType, aclOpExecutor *executor)
{
    auto otherTensor = executor->ConvertToTensor(other, dataType);
    return otherTensor;
}

aclnnStatus aclnnLogSpaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, double base, const aclTensor *result,
                                                uint64_t *workspaceSize, aclOpExecutor **executor){
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnLogSpace, DFX_IN(start, end, steps, base),DFX_OUT(result));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(start, end, steps, result);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    //如果steps为0，直接返回空张量
    if (steps == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    DataType result_dtype = result->GetDataType();

    // 推导数据类型，以输出类型为基准
    auto compute_dtype = result_dtype;

    // Linspace L0接口不支持int64输入，转为float32计算
    if (compute_dtype == DataType::DT_INT64) {
        compute_dtype = DataType::DT_FLOAT;
    }
    // start和end为FP16/BF16混合类型时升精度到FP32计算，避免Linspace和Pow多次cast导致精度损失
    if ((start->GetDataType() == DataType::DT_FLOAT16 && end->GetDataType() == DataType::DT_BF16) ||
        (start->GetDataType() == DataType::DT_BF16 && end->GetDataType() == DataType::DT_FLOAT16)) {
        compute_dtype = DataType::DT_FLOAT;
    }

    const aclTensor* start_tensor = ScalarToTensor(start, compute_dtype, uniqueExecutor.get());
    CHECK_RET(start_tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* end_tensor = ScalarToTensor(end, compute_dtype, uniqueExecutor.get());
    CHECK_RET(end_tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* linspace_result = l0op::Linspace(start_tensor, end_tensor, steps, uniqueExecutor.get());
    CHECK_RET(linspace_result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclScalar* base_scalar = uniqueExecutor.get()->AllocScalar(static_cast<float>(base));
    CHECK_RET(base_scalar != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* base_tensor = ScalarToTensor(base_scalar, compute_dtype, uniqueExecutor.get());
    CHECK_RET(base_tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* pow_result = l0op::InplacePow(base_tensor, linspace_result, uniqueExecutor.get());
    CHECK_RET(pow_result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果计算类型与输出类型不一致，Cast 转换后再 ViewCopy
    if (compute_dtype != result_dtype) {
        pow_result = l0op::Cast(pow_result, result_dtype, uniqueExecutor.get());
        CHECK_RET(pow_result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto view_copy_result = l0op::ViewCopy(pow_result, result, uniqueExecutor.get());
    CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogSpace(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream){
    L2_DFX_PHASE_2(aclnnLogSpace);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}