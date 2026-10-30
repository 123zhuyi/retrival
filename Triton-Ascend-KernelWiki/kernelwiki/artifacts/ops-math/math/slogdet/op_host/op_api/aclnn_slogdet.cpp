/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_slogdet.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "log_matrix_determinant.h"
#include "aclnn_kernels/reshape.h"

#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *signOut, const aclTensor *logOut) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(signOut, return false);
    OP_CHECK_NULL(logOut, return false);
    return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *signOut, const aclTensor *logOut) {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(signOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(logOut, DTYPE_SUPPORT_LIST, return false);
    if (IsComplexType(self->GetDataType()) && !IsComplexType(signOut->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The self's dtype is %s, signOut's dtype should also be complex type but got signOut with dtype %s.",
                op::ToString(self->GetDataType()).GetString(), op::ToString(signOut->GetDataType()).GetString());
        return false;
    }

    if (IsComplexType(self->GetDataType()) && !IsComplexType(logOut->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The self's dtype is %s, logOut's dtype should also be complex type but got signOut with dtype %s.",
                op::ToString(self->GetDataType()).GetString(), op::ToString(logOut->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *signOut, const aclTensor *logOut) {
    // 1、self >= 2维；2、shape是一组方阵（最后两维相等）3、signOut和logOut是self去掉最后两维
    auto dim = self->GetViewShape().GetDimNum();
    // self >= 2维
    OP_CHECK_MIN_DIM(self, 2, return false);
    // self是一组方阵，最后2维相等
    auto mDim = self->GetViewShape().GetDim(dim - 2);
    auto nDim = self->GetViewShape().GetDim(dim - 1);
    OP_CHECK(mDim == nDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "The last two dimensions of self must be equal, but they are %ld by %ld matrices.", mDim, nDim),
             return false);

    // sign和log是self去掉最后2维
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

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *signOut, const aclTensor *logOut) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, signOut, logOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, signOut, logOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足条件
    CHECK_RET(CheckShape(self, signOut, logOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ReshapeDim(const aclTensor *self, op::Shape &selfBatchShape, const aclTensor *&selfReshapeOut,
                              aclOpExecutor *executor) {
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

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor *self, aclTensor *signOut, aclTensor *logOut,
                                         uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnSlogdet, DFX_IN(self), DFX_OUT(signOut, logOut));
    // 参数检查
    auto ret = CheckParams(self, signOut, logOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *selfReshapeOut = nullptr;
    auto selfOriginalShape = self->GetViewShape();
    auto dim = selfOriginalShape.GetDimNum();
    op::Shape selfBatchShape;
    // 8维以上需要reshape到3维
    if (dim > MAX_SUPPORT_DIMS_NUMS) {
        ret = ReshapeDim(selfContiguous, selfBatchShape, selfReshapeOut, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        selfReshapeOut = selfContiguous;
    }

    auto logMatrixDeterminantOut = l0op::LogMatrixDeterminant(selfReshapeOut, uniqueExecutor.get());
    auto signValue = std::get<0>(logMatrixDeterminantOut);
    CHECK_RET(signValue != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto logValue = std::get<1>(logMatrixDeterminantOut);
    CHECK_RET(logValue != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 8维以上需要reshape
    const aclTensor *signReshapeOut = nullptr;
    const aclTensor *logReshapeOut = nullptr;
    if (dim > MAX_SUPPORT_DIMS_NUMS) {
        signReshapeOut = l0op::Reshape(signValue, selfBatchShape, uniqueExecutor.get());
        CHECK_RET(signReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        logReshapeOut = l0op::Reshape(logValue, selfBatchShape, uniqueExecutor.get());
        CHECK_RET(logReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        signReshapeOut = signValue;
        logReshapeOut = logValue;
    }
    auto signCastOut = l0op::Cast(signReshapeOut, signOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(signCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto logCastOut = l0op::Cast(logReshapeOut, logOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(logCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto signCopyResult = l0op::ViewCopy(signCastOut, signOut, uniqueExecutor.get());
    CHECK_RET(signCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto logCopyResult = l0op::ViewCopy(logCastOut, logOut, uniqueExecutor.get());
    CHECK_RET(logCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSlogdet(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnSlogdet);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
