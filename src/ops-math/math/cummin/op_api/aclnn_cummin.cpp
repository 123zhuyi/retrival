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
 * \file aclnn_cummin.cpp
 * \brief
 */
#include "aclnn_cummin.h"
#include "cummin.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32,   op::DataType::DT_INT64,  op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                          DataType::DT_INT8,  DataType::DT_UINT8,
                                                                          DataType::DT_INT32, op::DataType::DT_BF16};

// 根据dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* valuesOut, const aclTensor* indicesOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(valuesOut, SELF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(indicesOut, INDICES_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckDimValid(const aclTensor* self, const int64_t dim)
{
    const int64_t dimNum = self->GetViewShape().GetDimNum();
    int64_t minimum = dimNum * (-1);
    int64_t maximum = dimNum - 1;
    if (dimNum == 0 && (dim == 0 || dim == -1)) {
        return true;
    }
    if (dim < minimum || dim > maximum) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "dim must be within the range of [%ld, %ld], but it is %ld.", minimum, maximum,
            dim);
        return false;
    }
    return true;
}

static aclnnStatus CheckParamsCummin(
    const aclTensor* self, const int64_t dim, const aclTensor* valuesOut, const aclTensor* indicesOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull3Tensor(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查dim 是否合法
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShapeCumMinMax(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

inline static const aclIntArray* CalculateTranposePerm(const aclTensor* x, int64_t axis, aclOpExecutor* executor)
{
    auto dimSize = (int64_t)(x->GetViewShape().GetDimNum());
    std::vector<int64_t> perm(dimSize);
    if (axis < 0) {
        axis = axis + dimSize;
    }
    for (int64_t i = 0; i < dimSize; i++) {
        perm[i] = i;
    }
    std::swap(perm[axis], perm[0]);
    return executor->AllocIntArray(perm.data(), dimSize);
}

static aclnnStatus ExecCumminRegbase(
    const aclTensor* self, int64_t dim, aclTensor* valuesOut, aclTensor* indicesOut, aclOpExecutor* executor)
{
    auto selfCast = self;
    bool needCast = self->GetDataType() == op::DataType::DT_INT8 || self->GetDataType() == op::DataType::DT_UINT8;
    if (needCast) {
        selfCast = l0op::Cast(self, op::DataType::DT_FLOAT16, executor);
        CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    std::tuple<aclTensor*, aclTensor*> cumminResult;
    if (indicesOut->GetDataType() ==  op::DataType::DT_INT32) {
        cumminResult = l0op::CumminOutInt32(selfCast, dim, executor);
    } else {
        cumminResult = l0op::CumminOutInt64(selfCast, dim, executor);
    }
    

    const aclTensor* valuesResult = std::get<0>(cumminResult);
    CHECK_RET(valuesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* indicesResult = std::get<1>(cumminResult);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valuesCastResult = valuesResult;
    if (needCast) {
        valuesCastResult = l0op::Cast(valuesResult, valuesOut->GetDataType(), executor);
        CHECK_RET(valuesCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto valuesViewCopyResult = l0op::ViewCopy(valuesCastResult, valuesOut, executor);
    CHECK_RET(valuesViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesViewCopyResult = l0op::ViewCopy(indicesResult, indicesOut, executor);
    CHECK_RET(indicesViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecCummin(
    const aclTensor* self, int64_t dim, aclTensor* valuesOut, aclTensor* indicesOut, aclOpExecutor* executor)
{
    const aclIntArray* valuePerm = nullptr;
    int64_t realDim = dim;
    bool needTranspose = (dim != 0 && (IsAiCoreSupport(self) && self->GetViewShape().GetDim(dim) <= INT32_MAX));
    if (needTranspose) {
        valuePerm = CalculateTranposePerm(self, dim, executor);
        self = l0op::Transpose(self, valuePerm, executor);
        CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
        realDim = 0;
    }

    std::tuple<aclTensor*, aclTensor*> cumminResult;
    if (self->GetViewShape().GetDim(dim) > INT32_MAX ||
        (!IsAiCoreSupport(self) && indicesOut->GetDataType() == DataType::DT_INT64)) {
        cumminResult = l0op::CumminOutInt64(self, realDim, executor);
    } else {
        cumminResult = l0op::CumminOutInt32(self, realDim, executor);
    }
    const aclTensor* valuesResult = std::get<0>(cumminResult);
    CHECK_RET(valuesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* indicesResult = std::get<1>(cumminResult);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* valuesResultTranspose = valuesResult;
    const aclTensor* indicesResultTranspose = indicesResult;
    if (needTranspose) {
        valuesResultTranspose = l0op::Transpose(valuesResult, valuePerm, executor);
        CHECK_RET(valuesResultTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
        indicesResultTranspose = l0op::Transpose(indicesResult, valuePerm, executor);
        CHECK_RET(indicesResultTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto valuesCastResult = l0op::Cast(valuesResultTranspose, valuesOut->GetDataType(), executor);
    CHECK_RET(valuesCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto valuesViewCopyResult = l0op::ViewCopy(valuesCastResult, valuesOut, executor);
    CHECK_RET(valuesViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesCastResult = l0op::Cast(indicesResultTranspose, indicesOut->GetDataType(), executor);
    CHECK_RET(indicesCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesViewCopyResult = l0op::ViewCopy(indicesCastResult, indicesOut, executor);
    CHECK_RET(indicesViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static const aclTensor* AdaptInputZeroDimTensor(const aclTensor* self, aclOpExecutor* executor)
{
    if (self->GetViewShape().GetDimNum() != 0) {
        return self;
    }
    int64_t selfShapeValue[1] = {1};
    aclIntArray* selfShape = executor->AllocIntArray(selfShapeValue, 1);
    CHECK_RET(selfShape != nullptr, nullptr);
    auto selfReshape = l0op::Reshape(self, selfShape, executor);
    CHECK_RET(selfReshape != nullptr, nullptr);
    return selfReshape;
}

static void CheckFormat(const aclTensor* self)
{
    ge::Format selfStorageFormat = self->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnCummin doesn't support format NZ.");
    }
}

aclnnStatus aclnnCumminGetWorkspaceSize(
    const aclTensor* self, int64_t dim, aclTensor* valuesOut, aclTensor* indicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnCummin, DFX_IN(self, dim), DFX_OUT(valuesOut, indicesOut));

    auto ret = CheckParamsCummin(self, dim, valuesOut, indicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CheckFormat(self);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // process 0_d tensor
    auto selfReshapeContinuous = AdaptInputZeroDimTensor(selfContiguous, uniqueExecutor.get());
    CHECK_RET(selfReshapeContinuous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (selfReshapeContinuous->GetDataType() == DataType::DT_BOOL) {
        selfReshapeContinuous = l0op::Cast(selfReshapeContinuous, DataType::DT_UINT8, uniqueExecutor.get());
        CHECK_RET(selfReshapeContinuous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch)) {
        ret = ExecCumminRegbase(selfReshapeContinuous, dim, valuesOut, indicesOut, uniqueExecutor.get());
    } else {
        ret = ExecCummin(selfReshapeContinuous, dim, valuesOut, indicesOut, uniqueExecutor.get());
    }

    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCummin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCummin);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
