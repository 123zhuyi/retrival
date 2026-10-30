/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_prod.h"
#include "reduce_prod.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "conversion/fill/op_api/fill.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t MAX_DIM = 8;
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,   op::DataType::DT_DOUBLE,    op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT16,     op::DataType::DT_INT32,     op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,   op::DataType::DT_DOUBLE,     op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT16,     op::DataType::DT_INT32,      op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> EMPTY_INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclDataType dtype, const aclTensor* out)
{
    // 检查输入self的数据类型是否在ReduceProd算子的支持列表内
    bool isASCEND910B =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 || IsRegBase());
    if (isASCEND910B) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    // 检查输出out的dtype与参数dtype是否相同
    OP_CHECK_DTYPE_NOT_MATCH(out, op::ToOpDataType(dtype), return false);

    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM, return false);
    return true;
}

static bool CheckDimValid(const aclTensor* self, int64_t dim, bool isCheckDim)
{
    if (!isCheckDim) {
        return true;
    }
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    // 检查指定dim是否在self的维度范围内
    if (selfDimNum == 0){
        if (dim == 0 || dim == -1) {
            return true;
        }
    }

    if (dim >= selfDimNum || dim < (-selfDimNum)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected dim to be in range of [%ld, %ld], but got %ld.", -selfDimNum,
            selfDimNum - 1, dim);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(
    const aclTensor* self, int64_t dim, const aclDataType dtype, const aclTensor* out, bool isCheckDim)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否是否有效
    CHECK_RET(CheckDtypeValid(self, dtype, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入tensor的shape是否异常，输出和输入的shape是否相同
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查reduce的轴是否超出self维度范围
    CHECK_RET(CheckDimValid(self, dim, isCheckDim), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(aclTensor* out, float val, aclOpExecutor* executor)
{
    if (!CheckType(out->GetDataType(), EMPTY_INPUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when input is empty tensor, out cann not be %s, should be in dtype support list %s.",
            ToString(out->GetDataType()).GetString(), ToString(EMPTY_INPUT_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    FVector<int64_t> shape;
    size_t dimNum = out->GetViewShape().GetDimNum();
    for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = out->GetViewShape().GetDim(idx);
        if (tmpVal == 0) {
            return ACLNN_SUCCESS;
        }
        shape.push_back(tmpVal);
    }
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecuteProd(
    const aclTensor* self, const aclTensor* axes, bool keepDim, const aclDataType dtype, aclTensor* out,
    aclOpExecutor* executor)
{
    // 将输入tensor转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // ReduceProd算子不支持bool类型计算，将bool转为float
    auto selfCast = [&dtype, &selfContiguous, &out, &executor]() -> const aclTensor* {
        if (IsRegBase()) {
            if (selfContiguous->GetDataType() == op::DataType::DT_FLOAT16 &&
                op::ToOpDataType(dtype) == op::DataType::DT_FLOAT16) {
                return l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, executor);
            }
            if ((selfContiguous->GetDataType() == op::DataType::DT_BF16 ||
                 selfContiguous->GetDataType() == op::DataType::DT_FLOAT16) &&
                out->GetDataType() == op::DataType::DT_FLOAT) {
                return selfContiguous;
            }
            if (selfContiguous->GetDataType() == op::DataType::DT_UINT8 &&
                op::ToOpDataType(dtype) == op::DataType::DT_BOOL) {
                auto tmpTensor = const_cast<aclTensor*>(selfContiguous);
                tmpTensor->SetDataType(op::DataType::DT_INT8);
                selfContiguous = tmpTensor;
                return l0op::Cast(selfContiguous, out->GetDataType(), executor);
            }
            return l0op::Cast(selfContiguous, out->GetDataType(), executor);
        } else {
            if (out->GetDataType() == op::DataType::DT_BOOL ||
                (selfContiguous->GetDataType() == op::DataType::DT_FLOAT16 &&
                 op::ToOpDataType(dtype) == op::DataType::DT_FLOAT16)) {
                return l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, executor);
            }
            return l0op::Cast(selfContiguous, out->GetDataType(), executor);
        }
    }();
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用ReduceProd算子kernel，完成计算
    auto output = l0op::ReduceProd(selfCast, axes, keepDim, executor);
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckShapeAndScalarSame(output, out), ACLNN_ERR_PARAM_INVALID);

    // 将输出转换为指定的类型
    auto outCasted = l0op::Cast(output, out->GetDataType(), executor);
    CHECK_RET(outCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto copyResult = l0op::ViewCopy(outCasted, out, executor);
    CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnProdDimGetWorkspaceSize(
    const aclTensor* self, int64_t dim, bool keepDim, const aclDataType dtype, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnProdDim, DFX_IN(self, dim, keepDim, dtype), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查，此接口中需要检查dimList是否为nullptr
    auto ret = CheckParams(self, dim, dtype, out, true);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入为空tensor时，输出是值为1的tensor
    if (self->IsEmpty()) {
        // 空tensor填充1
        ret = FillScalar(out, 1.0f, uniqueExecutor.get());
        if (ret == ACLNN_SUCCESS) {
            *workspaceSize = uniqueExecutor->GetWorkspaceSize();
            uniqueExecutor.ReleaseTo(executor);
        }
        return ret;
    }

    auto axes = uniqueExecutor.get()->ConvertToTensor(&dim, 1, op::DataType::DT_INT64);
    CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ret = ExecuteProd(self, axes, keepDim, dtype, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnProdGetWorkspaceSize(
    const aclTensor* self, const aclDataType dtype, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnProd, DFX_IN(self, dtype), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查，此接口中不需要检查dimList是否为nullptr
    auto ret = CheckParams(self, 0, dtype, out, false);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入为空tensor时，输出是值为1的tensor
    if (self->IsEmpty()) {
        // 空tensor填充1
        ret = FillScalar(out, 1.0f, uniqueExecutor.get());
        if (ret == ACLNN_SUCCESS) {
            *workspaceSize = uniqueExecutor->GetWorkspaceSize();
            uniqueExecutor.ReleaseTo(executor);
        }
        return ret;
    }

    // 根据self，获取dimList
    size_t dimNum = self->GetViewShape().GetDimNum();
    int64_t sizes[dimNum];
    for (size_t i = 0; i < dimNum; ++i) {
        sizes[i] = i;
    }
    aclIntArray* dimList = uniqueExecutor.get()->AllocIntArray(sizes, dimNum);
    CHECK_RET(dimList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 创建axesTensor
    auto axes = uniqueExecutor.get()->ConvertToTensor(dimList, op::DataType::DT_INT64);
    CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行计算图逻辑
    ret = ExecuteProd(self, axes, false, dtype, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnProdDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnProdDim);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnProd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnProd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
