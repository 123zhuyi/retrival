/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_any.h"
#include "aclnn_any.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "conversion/fill/op_api/fill.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_BOOL,    op::DataType::DT_DOUBLE};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT8,
    op::DataType::DT_INT16,  op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<DataType> NON_CONTIOUS_SUPPORT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static const std::initializer_list<DataType> out_dtype_support_list = {op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static bool CheckNotNull(const aclTensor* self, const aclIntArray* dim, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(dim, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static const aclIntArray* getAllDims(const aclTensor* self, aclOpExecutor* executor)
{
    auto input_shape = self->GetViewShape();
    const size_t input_dim_num = input_shape.GetDimNum();
    std::vector<int64_t> dims(input_dim_num);
    for (size_t idx = 0; idx < input_dim_num; idx++) {
        dims[idx] = idx;
    }
    return executor->AllocIntArray(dims.data(), input_dim_num);
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 || IsRegBase()) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    const auto& supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, out_dtype_support_list, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);

    return true;
}

static bool CheckDim(const aclTensor* self, const aclIntArray* dim)
{
    auto input_shape = self->GetViewShape();
    int64_t input_dim_num = input_shape.GetDimNum();
    // self为scalar时，其dim视为1维
    if (input_dim_num == 0) {
        input_dim_num = 1;
    }
    for (size_t idx = 0; idx < dim->Size(); idx++) {
        if ((*dim)[idx] < -(input_dim_num) || (*dim)[idx] >= input_dim_num) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range.");
            return false;
        }
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, dim, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* GetTensorWithValueFalse(aclTensor* out, aclOpExecutor* executor)
{
    if (out->IsEmpty()) {
        return out;
    }
    aclScalar* scalar = executor->AllocScalar(0);
    auto valueTensor = executor->ConvertToTensor(scalar, out->GetDataType());
    auto outputDims = op::ToShapeVector(out->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(outputDims.data(), outputDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    auto falseTensor = l0op::Fill(dimTensor, valueTensor, dimArray, executor);
    if (falseTensor == nullptr) {
        return nullptr;
    }
    auto viewCopyResult = l0op::ViewCopy(falseTensor, out, executor);
    return viewCopyResult;
}

static bool IsNonContiguousSupport(const aclTensor* self, const aclIntArray* dim)
{
    if (!IsRegBase()) {
        return false;
    }
    if (!CheckType(self->GetDataType(), NON_CONTIOUS_SUPPORT_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    if (!op::IsReduceNonContiguousSupport(self, dim)) {
        return false;
    }
    return true;
}

aclnnStatus aclnnAnyGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAny, DFX_IN(self, dim, keepdim), DFX_OUT(out));

    auto ret = CheckParams(self, dim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空dim处理
    if (dim->Size() == 0) {
        dim = getAllDims(self, uniqueExecutor.get());
        CHECK_RET(dim != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (self->IsEmpty()) {
        auto falseTensor = GetTensorWithValueFalse(out, uniqueExecutor.get());
        CHECK_RET(falseTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (IsNonContiguousSupport(self, dim)) {
        OP_LOGI("Entering NonContiguous Reduce Any");
        auto selfContiguous = uniqueExecutor.get()->CreateView(
            self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto anyResult = l0op::ReduceAny(selfContiguous, dim, keepdim, uniqueExecutor.get());
        CHECK_RET(anyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckShapeAndScalarSame(anyResult, out), ACLNN_ERR_PARAM_INVALID);

        auto anyResultCasted = l0op::Cast(anyResult, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(anyResultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto anyResultFormat = l0op::ReFormat(anyResultCasted, out->GetStorageFormat());
        CHECK_RET(anyResultFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(anyResultFormat, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        OP_LOGI("Entering Contiguous Reduce Any");
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto selfCasted = selfContiguous;
        if (!((selfContiguous->GetDataType() == op::DataType::DT_FLOAT16 || selfContiguous->GetDataType() == op::DataType::DT_BF16 || selfContiguous->GetDataType() == op::DataType::DT_FLOAT) &&
                (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B || GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93))) {
            // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
            selfCasted = l0op::Cast(selfContiguous, DataType::DT_BOOL, uniqueExecutor.get());
            CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        auto anyResult = l0op::ReduceAny(selfCasted, dim, keepdim, uniqueExecutor.get());
        CHECK_RET(anyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckShapeAndScalarSame(anyResult, out), ACLNN_ERR_PARAM_INVALID);

        auto anyResultCasted = l0op::Cast(anyResult, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(anyResultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto anyResultFormat = l0op::ReFormat(anyResultCasted, out->GetStorageFormat());
        CHECK_RET(anyResultFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(anyResultFormat, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAny(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAny);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}