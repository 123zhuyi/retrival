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
 * \file aclnn_div_v3.cpp
 * \brief DivV3 L2 API (aclnn) implementation with broadcast support
 */

#include "aclnn_div_v3.h"
#include "div_v3.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32, DataType::DT_INT16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    if (!CheckType(self->GetDataType(), DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self dtype %s not in support list.",
            op::ToString(self->GetDataType()).GetString());
        return false;
    }
    if (self->GetDataType() != other->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and other must have same dtype.");
        return false;
    }
    if (self->GetDataType() != out->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input and output must have same dtype.");
        return false;
    }
    return true;
}

static bool CheckModeValid(int64_t mode)
{
    if (mode < 0 || mode > 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode must be 0 (RealDiv), 1 (TruncDiv), or 2 (FloorDiv), but got %ld.", mode);
        return false;
    }
    return true;
}

static bool CheckBroadcastShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(other, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    OP_CHECK_BROADCAST(self, other, return false);
    Shape broadcastShape;
    BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape);

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "broadcast shape %s != out shape %s.", op::ToString(broadcastShape).GetString(),
            op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, int64_t mode, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckModeValid(mode), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckBroadcastShape(self, other, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclIntArray* GetShapeAsIntArray(const aclTensor* tensor, aclOpExecutor* executor)
{
    int64_t dimNum = static_cast<int64_t>(tensor->GetViewShape().GetDimNum());
    if (dimNum == 0) {
        int64_t shape[1] = {1};
        return executor->AllocIntArray(shape, 1);
    }
    std::vector<int64_t> shape(dimNum);
    for (int64_t i = 0; i < dimNum; i++) {
        shape[i] = tensor->GetViewShape()[i];
    }
    return executor->AllocIntArray(shape.data(), dimNum);
}

aclnnStatus aclnnDivV3GetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, int64_t mode, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDivV3, DFX_IN(self, other), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, other, mode, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || other->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 1. make contiguous
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 2. broadcast to output shape
    auto outShape = GetShapeAsIntArray(out, uniqueExecutor.get());
    CHECK_RET(outShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (selfContiguous->GetViewShape() != out->GetViewShape()) {
        selfContiguous = l0op::BroadcastTo(selfContiguous, outShape, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (otherContiguous->GetViewShape() != out->GetViewShape()) {
        otherContiguous = l0op::BroadcastTo(otherContiguous, outShape, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 3. call DivV3 kernel
    auto opResult = l0op::DivV3(selfContiguous, otherContiguous, mode, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 4. copy result to output
    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDivV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDivV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
