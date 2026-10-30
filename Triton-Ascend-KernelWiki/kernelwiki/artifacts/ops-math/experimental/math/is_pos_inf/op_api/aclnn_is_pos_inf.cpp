/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_is_pos_inf.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/fill/op_api/fill.h"
#include "is_pos_inf.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16, DataType::DT_BOOL, DataType::DT_INT32,
    DataType::DT_INT64,   DataType::DT_INT16, DataType::DT_INT8, DataType::DT_UINT8};

static bool CheckSocAndDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto soc = GetCurrentPlatformInfo().GetSocVersion();
    if (soc < SocVersion::ASCEND910B || soc > SocVersion::ASCEND910E) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnIsPosInf is only supported on Ascend 910B/910C class devices.");
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, DataType::DT_BOOL, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSocAndDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static FVector<int64_t> GetTensorShapeVec(const aclTensor* tensor)
{
    FVector<int64_t> dims;
    if (tensor->GetViewShape().GetDimNum() == 0) {
        dims.push_back(1);
        return dims;
    }
    for (size_t idx = 0; idx < tensor->GetViewShape().GetDimNum(); ++idx) {
        dims.push_back(tensor->GetViewShape().GetDim(idx));
    }
    return dims;
}

static const aclTensor* FillBoolTensor(aclTensor* out, bool val, aclOpExecutor* executor)
{
    auto dimsVec = GetTensorShapeVec(out);
    const aclTensor* dimsTensor = executor->ConvertToTensor(dimsVec.data(), dimsVec.size(), ToOpDataType(ACL_INT64));
    aclIntArray* shapeArray = executor->AllocIntArray(dimsVec.data(), dimsVec.size());
    FVector<bool> valVec = {val};
    const aclTensor* valTensor = executor->ConvertToTensor(valVec.data(), valVec.size(), out->GetDataType());
    if (dimsTensor == nullptr || shapeArray == nullptr || valTensor == nullptr) {
        return nullptr;
    }
    return l0op::Fill(dimsTensor, valTensor, shapeArray, executor);
}

aclnnStatus aclnnIsPosInfGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnIsPosInf, DFX_IN(self), DFX_OUT(out));
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* result = nullptr;
    if (!IsFloatingType(self->GetDataType()) && !IsComplexType(self->GetDataType())) {
        result = FillBoolTensor(out, false, uniqueExecutor.get());
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        result = l0op::IsPosInf(selfContiguous, uniqueExecutor.get());
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIsPosInf(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIsPosInf);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
