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
 * @file aclnn_roll.cpp
 * @brief ACLNN Roll implementation.
 */

#include "aclnn_roll.h"
#include "roll.h"

#include <algorithm>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace {
constexpr size_t MAX_SUPPORT_DIMS_NUMS = 8;

const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,   op::DataType::DT_INT8,  op::DataType::DT_BF16,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_UINT32};

bool CheckNotNull(const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out)
{
    (void)dims;
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(shifts, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

bool IsDtypeSupported(op::DataType dtype)
{
    return std::find(DTYPE_SUPPORT_LIST.begin(), DTYPE_SUPPORT_LIST.end(), dtype) != DTYPE_SUPPORT_LIST.end();
}

bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    if (!IsDtypeSupported(x->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x dtype is not supported.");
        return false;
    }
    if (x->GetDataType() != out->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x and output out must have the same dtype.");
        return false;
    }
    return true;
}

bool CheckFormatValid(const aclTensor* x, const aclTensor* out)
{
    constexpr auto supportedFormat = op::Format::FORMAT_ND;
    if (x->GetViewFormat() != supportedFormat || x->GetStorageFormat() != supportedFormat) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x only supports ND format.");
        return false;
    }
    if (out->GetViewFormat() != supportedFormat || out->GetStorageFormat() != supportedFormat) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output out only supports ND format.");
        return false;
    }
    return true;
}

bool CheckShapeValid(const aclTensor* x, const aclTensor* out)
{
    if (x->GetViewShape() != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x and output out must have the same shape.");
        return false;
    }
    if (x->GetViewShape().GetDimNum() > MAX_SUPPORT_DIMS_NUMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Roll supports at most 8 dims.");
        return false;
    }
    return true;
}

bool CheckArraySize(const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims)
{
    const auto tensorDim = x->GetViewShape().GetDimNum();
    const size_t dimsSize = dims == nullptr ? 0U : dims->Size();
    if (tensorDim == 0) {
        if (dimsSize != 0U || shifts->Size() != 1U) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For 0-D tensor, shifts size must be 1 and dims must be empty.");
            return false;
        }
        return true;
    }
    if (dimsSize == 0U) {
        if (shifts->Size() != 1U) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When dims is empty, shifts size must be 1.");
            return false;
        }
        return true;
    }
    if (shifts->Size() != dimsSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shifts and dims must have the same size.");
        return false;
    }
    return true;
}

bool CheckDimsRange(const aclTensor* x, const aclIntArray* dims)
{
    if (dims == nullptr) {
        return true;
    }
    const int64_t tensorDim = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    for (size_t i = 0; i < dims->Size(); ++i) {
        const int64_t dim = (*dims)[i];
        if (dim < -tensorDim || dim >= tensorDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dims value is out of range.");
            return false;
        }
    }
    return true;
}

bool HasDenseViewLayout(const aclTensor* tensor)
{
    if (tensor == nullptr || tensor->GetViewOffset() != 0) {
        return false;
    }
    const auto& viewShape = tensor->GetViewShape();
    const auto& strides = tensor->GetViewStrides();
    if (strides.size() != viewShape.GetDimNum()) {
        return false;
    }
    int64_t expectedStride = 1;
    for (int64_t i = static_cast<int64_t>(viewShape.GetDimNum()) - 1; i >= 0; --i) {
        if (strides[static_cast<size_t>(i)] != expectedStride) {
            return false;
        }
        expectedStride *= static_cast<int64_t>(viewShape.GetDim(static_cast<size_t>(i)));
    }
    return true;
}

bool CanWriteOutDirectly(const aclTensor* tensor)
{
    if (!HasDenseViewLayout(tensor)) {
        return false;
    }
    const auto& storageShape = tensor->GetStorageShape();
    const auto& viewShape = tensor->GetViewShape();
    return storageShape.GetDimNum() == 0 || storageShape == viewShape;
}

aclTensor* NormalizeEmptyStorageTensor(const aclTensor* tensor, aclOpExecutor* executor)
{
    if (tensor == nullptr || executor == nullptr || tensor->GetStorageShape().GetDimNum() != 0 ||
        tensor->GetViewShape().GetDimNum() == 0) {
        return const_cast<aclTensor*>(tensor);
    }
    return executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
}

aclnnStatus CheckParams(const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(x, shifts, dims, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(x, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(x, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckArraySize(x, shifts, dims), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDimsRange(x, dims), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" aclnnStatus aclnnRollGetWorkspaceSize(const aclTensor* x,
                                                 const aclIntArray* shifts,
                                                 const aclIntArray* dims,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnRoll, DFX_IN(x, shifts, dims), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, shifts, dims, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* xForRoll = NormalizeEmptyStorageTensor(x, uniqueExecutor.get());
    CHECK_RET(xForRoll != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor* outForRoll = NormalizeEmptyStorageTensor(out, uniqueExecutor.get());
    CHECK_RET(outForRoll != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* xContiguous = xForRoll;
    if (!HasDenseViewLayout(x)) {
        xContiguous = l0op::Contiguous(xForRoll, uniqueExecutor.get());
        CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* rollResult = nullptr;
    if (CanWriteOutDirectly(outForRoll)) {
        rollResult = l0op::Roll(xContiguous, shifts, dims, outForRoll, uniqueExecutor.get());
    } else {
        rollResult = l0op::Roll(xContiguous, shifts, dims, uniqueExecutor.get());
        CHECK_RET(rollResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        rollResult = l0op::ViewCopy(rollResult, out, uniqueExecutor.get());
    }
    CHECK_RET(rollResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnRoll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoll);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
