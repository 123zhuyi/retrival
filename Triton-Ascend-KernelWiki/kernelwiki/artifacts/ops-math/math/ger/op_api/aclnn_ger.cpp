/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ger.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "ger.h"
#include "math/mul/op_api/mul.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "math/logical_and/op_api/logical_and.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t AXIS_DIM = 1;
static constexpr int64_t EXPECT_SIZE = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> GER_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> ASCEND950_GER_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,   op::DataType::DT_INT32,     op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,     op::DataType::DT_INT16,     op::DataType::DT_BOOL,
    op::DataType::DT_INT64,     op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,   op::DataType::DT_INT32,     op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,     op::DataType::DT_INT16,     op::DataType::DT_BOOL,
    op::DataType::DT_BF16,  op::DataType::DT_INT64,     op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* vec2, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(vec2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* vec2)
{
     if (IsRegBase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND950_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(vec2, ASCEND950_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(vec2, DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckPromoteType(
    const op::DataType selfDtype, const op::DataType vec2Dtype, const op::DataType outDtype,
    const op::DataType promoteType)
{
    // 检查self和vec2能否做数据类型推导
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and vec2 dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(vec2Dtype).GetString());
        return false;
    }

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* vec2, const aclTensor* out)
{
    size_t selfDimNum = self->GetViewShape().GetDimNum();
    size_t vec2DimNum = vec2->GetViewShape().GetDimNum();
    size_t outDimNum = out->GetViewShape().GetDimNum();
    OP_CHECK(
        selfDimNum == 1, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected 1-D argument, but got %zu-D.", selfDimNum),
        return false);

    OP_CHECK(
        vec2DimNum == 1, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected 1-D argument, but got %zu-D.", vec2DimNum),
        return false);

    OP_CHECK(
        outDimNum == EXPECT_SIZE, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected 2-D argument, but got %zu-D.", outDimNum),
        return false);
    int64_t size0 = out->GetViewShape().GetDim(0);
    int64_t size1 = out->GetViewShape().GetDim(1);
    OP_CHECK(
        size0 == self->GetViewShape().GetDim(0) && size1 == vec2->GetViewShape().GetDim(0),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected sizes of out{%ld, %ld}"
            "should be equal to self * vec2 {%ld, %ld}",
            size0, size1, self->GetViewShape().GetDim(0), vec2->GetViewShape().GetDim(0)),
        return false);

    return true;
}

static inline aclnnStatus CheckParams(const aclTensor* self, const aclTensor* vec2, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, vec2, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, vec2), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和vec2能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    op::DataType promoteType = op::PromoteType(self->GetDataType(), vec2->GetDataType());
    CHECK_RET(
        CheckPromoteType(self->GetDataType(), vec2->GetDataType(), out->GetDataType(), promoteType),
        ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出shape
    CHECK_RET(CheckShape(self, vec2, out), ACLNN_ERR_PARAM_INVALID);
    
    if (self->GetStorageFormat() != Format::FORMAT_ND || vec2->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGW("Only support ND format for ger operator.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGerGetWorkspaceSize(
    const aclTensor* self, const aclTensor* vec2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnGer, DFX_IN(self, vec2), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    OP_CHECK(
        uniqueExecutor.get() != nullptr, OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Create executor error."),
        return ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, vec2, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || vec2->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto promoteType = op::PromoteType(self->GetDataType(), vec2->GetDataType());

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto vec2Contiguous = l0op::Contiguous(vec2, uniqueExecutor.get());
    CHECK_RET(vec2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入vec2的数据类型转换成隐式数据类型
    auto vec2Casted = l0op::Cast(vec2Contiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(vec2Casted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* calcOut = nullptr;
    
    // 950版本支持bf16 fp16 fp32 走Ger接口；
    // 其他版本支持fp16 fp32 走Ger接口
    if ((IsRegBase() && CheckType(selfCasted->GetDataType(), ASCEND950_GER_SUPPORT_LIST)) ||
        (!IsRegBase() && CheckType(selfCasted->GetDataType(), GER_SUPPORT_LIST))) {
        calcOut = l0op::Ger(selfCasted, vec2Casted, uniqueExecutor.get());
    } else {
        auto selfNd = l0op::UnsqueezeNd(selfCasted, AXIS_DIM, uniqueExecutor.get());
        CHECK_RET(selfNd != nullptr, ACLNN_ERR_INNER_NULLPTR);

        calcOut = selfNd->GetDataType() == op::DataType::DT_BOOL ?
                    l0op::LogicalAnd(selfNd, vec2Casted, uniqueExecutor.get()) :
                    l0op::Mul(selfNd, vec2Casted, uniqueExecutor.get());
    }

    CHECK_RET(calcOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto castOut = l0op::Cast(calcOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGer(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGer);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
