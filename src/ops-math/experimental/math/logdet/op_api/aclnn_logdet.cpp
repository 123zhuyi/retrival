/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_logdet.h"

#include "logdet.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

// 仅 fp32
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // 1. self rank >= 2; 2. last two dims form square matrices; 3. out shape is self.shape[:-2].
    auto dim = self->GetViewShape().GetDimNum();
    OP_CHECK_MIN_DIM(self, 2, return false);

    auto mDim = self->GetViewShape().GetDim(dim - 2);
    auto nDim = self->GetViewShape().GetDim(dim - 1);
    OP_CHECK(mDim == nDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "The last two dimensions of self must be equal, but they are %ld by %ld matrices.", mDim, nDim),
             return false);

    auto selfBatchShapeVec = ToShapeVector(self->GetViewShape());
    selfBatchShapeVec.pop_back();
    selfBatchShapeVec.pop_back();
    op::Shape selfBatchShape;
    ToShape(selfBatchShapeVec, selfBatchShape);
    OP_CHECK(out->GetViewShape() == selfBatchShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect shape of out is %s, but got %s.",
                     op::ToString(selfBatchShape).GetString(), op::ToString(out->GetViewShape()).GetString()),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ReshapeDim(const aclTensor* self, op::Shape& selfBatchShape, const aclTensor*& selfReshapeOut,
                              aclOpExecutor* executor)
{
    auto selfOriginalShape = self->GetViewShape();
    auto dim = self->GetViewShape().GetDimNum();
    auto lastDim = self->GetViewShape().GetDim(dim - 1);
    auto newDim = self->Numel() / (lastDim * lastDim);
    op::Shape selfNewShape = {newDim, lastDim, lastDim};
    selfReshapeOut = l0op::Reshape(self, selfNewShape, executor);
    CHECK_RET(selfReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto shapeVec = ToShapeVector(selfOriginalShape);
    shapeVec.pop_back();
    shapeVec.pop_back();
    ToShape(shapeVec, selfBatchShape);
    return ACLNN_SUCCESS;
}

// 计算主链：Contiguous → reshape → l0op::Logdet → reshape 回 → Cast → ViewCopy。
static aclnnStatus LogdetExecImpl(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* selfReshapeOut = nullptr;
    auto selfOriginalShape = self->GetViewShape();
    auto dim = selfOriginalShape.GetDimNum();
    op::Shape selfBatchShape;
    // 8 维以上需要 reshape 到 3 维（host 侧归一，kernel 透明）
    if (dim > MAX_SUPPORT_DIMS_NUMS) {
        auto ret = ReshapeDim(selfContiguous, selfBatchShape, selfReshapeOut, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        selfReshapeOut = selfContiguous;
    }

    auto logdetOut = l0op::Logdet(selfReshapeOut, executor);
    CHECK_RET(logdetOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 8 维以上需要 reshape 回原 batch 形状
    const aclTensor* logdetReshapeOut = nullptr;
    if (dim > MAX_SUPPORT_DIMS_NUMS) {
        logdetReshapeOut = l0op::Reshape(logdetOut, selfBatchShape, executor);
        CHECK_RET(logdetReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        logdetReshapeOut = logdetOut;
    }

    auto logdetCastOut = l0op::Cast(logdetReshapeOut, out->GetDataType(), executor);
    CHECK_RET(logdetCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // out 可能非连续，结果通过 ViewCopy 写回。
    auto logdetCopyResult = l0op::ViewCopy(logdetCastOut, out, executor);
    CHECK_RET(logdetCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnLogdetGetWorkspaceSize(const aclTensor* self, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnLogdet, DFX_IN(self), DFX_OUT(out));

    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空 Tensor 边界：workspaceSize=0，直接返回成功
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    ret = LogdetExecImpl(self, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogdet(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLogdet);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
