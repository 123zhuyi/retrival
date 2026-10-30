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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "aclnn_slogdet_native.h"

#include "slogdet_api.h"
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

// 仅 fp32（实验树原生实现 CP1 锁定）
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* signOut, const aclTensor* logOut)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(signOut, return false);
    OP_CHECK_NULL(logOut, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* signOut, const aclTensor* logOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(signOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(logOut, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* signOut, const aclTensor* logOut)
{
    // 1、self >= 2 维；2、shape 是一组方阵（末两维相等）；3、signOut/logOut 是 self 去掉末两维
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
    OP_CHECK(signOut->GetViewShape() == selfBatchShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect shape of signOut is %s, but got %s.",
                     op::ToString(selfBatchShape).GetString(), op::ToString(signOut->GetViewShape()).GetString()),
             return false);
    OP_CHECK(logOut->GetViewShape() == selfBatchShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect shape of logOut is %s, but got %s.",
                     op::ToString(selfBatchShape).GetString(), op::ToString(logOut->GetViewShape()).GetString()),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* signOut, const aclTensor* logOut)
{
    CHECK_RET(CheckNotNull(self, signOut, logOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, signOut, logOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, signOut, logOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ReshapeDim(const aclTensor* self, op::Shape& selfBatchShape, const aclTensor*& selfReshapeOut,
                              aclOpExecutor* executor)
{
    auto selfOriginalShape = self->GetViewShape();
    auto dim = self->GetViewShape().GetDimNum();
    auto lastDim = self->GetViewShape().GetDim(dim - 1);
    auto newDim = self->Size() / (lastDim * lastDim);
    op::Shape selfNewShape = {newDim, lastDim, lastDim};
    selfReshapeOut = l0op::Reshape(self, selfNewShape, executor);
    CHECK_RET(selfReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto shapeVec = ToShapeVector(selfOriginalShape);
    shapeVec.pop_back();
    shapeVec.pop_back();
    ToShape(shapeVec, selfBatchShape);
    return ACLNN_SUCCESS;
}

static aclnnStatus NormalizeSlogdetInput(const aclTensor* self, aclOpExecutor* executor,
                                         const aclTensor*& selfReshapeOut, op::Shape& selfBatchShape,
                                         bool& needBatchReshape)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dim = self->GetViewShape().GetDimNum();
    needBatchReshape = dim > MAX_SUPPORT_DIMS_NUMS;
    if (needBatchReshape) {
        return ReshapeDim(selfContiguous, selfBatchShape, selfReshapeOut, executor);
    }

    selfReshapeOut = selfContiguous;
    return ACLNN_SUCCESS;
}

static aclnnStatus RestoreSlogdetBatchShape(const aclTensor* signValue, const aclTensor* logValue,
                                            const op::Shape& selfBatchShape, bool needBatchReshape,
                                            aclOpExecutor* executor, const aclTensor*& signReshapeOut,
                                            const aclTensor*& logReshapeOut)
{
    if (!needBatchReshape) {
        signReshapeOut = signValue;
        logReshapeOut = logValue;
        return ACLNN_SUCCESS;
    }

    signReshapeOut = l0op::Reshape(signValue, selfBatchShape, executor);
    CHECK_RET(signReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    logReshapeOut = l0op::Reshape(logValue, selfBatchShape, executor);
    CHECK_RET(logReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CastAndCopySlogdetOutputs(const aclTensor* signValue, const aclTensor* logValue, aclTensor* signOut,
                                             aclTensor* logOut, aclOpExecutor* executor)
{
    auto signCastOut = l0op::Cast(signValue, signOut->GetDataType(), executor);
    CHECK_RET(signCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto logCastOut = l0op::Cast(logValue, logOut->GetDataType(), executor);
    CHECK_RET(logCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto signCopyResult = l0op::ViewCopy(signCastOut, signOut, executor);
    CHECK_RET(signCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto logCopyResult = l0op::ViewCopy(logCastOut, logOut, executor);
    CHECK_RET(logCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor* self, aclTensor* signOut, aclTensor* logOut,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSlogdet, DFX_IN(self), DFX_OUT(signOut, logOut));

    auto ret = CheckParams(self, signOut, logOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空 Tensor 边界：workspaceSize=0，直接返回成功
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* selfReshapeOut = nullptr;
    op::Shape selfBatchShape;
    bool needBatchReshape = false;
    ret = NormalizeSlogdetInput(self, uniqueExecutor.get(), selfReshapeOut, selfBatchShape, needBatchReshape);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto slogdetOut = l0op::Slogdet(selfReshapeOut, uniqueExecutor.get());
    auto signValue = std::get<0>(slogdetOut);
    CHECK_RET(signValue != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto logValue = std::get<1>(slogdetOut);
    CHECK_RET(logValue != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* signReshapeOut = nullptr;
    const aclTensor* logReshapeOut = nullptr;
    ret = RestoreSlogdetBatchShape(signValue, logValue, selfBatchShape, needBatchReshape, uniqueExecutor.get(),
                                   signReshapeOut, logReshapeOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CastAndCopySlogdetOutputs(signReshapeOut, logReshapeOut, signOut, logOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSlogdet(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSlogdet);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
