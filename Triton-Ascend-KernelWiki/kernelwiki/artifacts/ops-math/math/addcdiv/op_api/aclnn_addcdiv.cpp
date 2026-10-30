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
 * \file aclnn_addcdiv.cpp
 * \brief
 */
#include "aclnn_addcdiv.h"
#include "addcdiv.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_SUPPORT_DIM = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16};

static bool CheckNotNull(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
    const aclTensor* out)
{
    // 校验输入输出是否为空
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(tensor1, return false);
    OP_CHECK_NULL(tensor2, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static op::DataType GetScalarDefaultDtype(const op::DataType input)
{
    if (IsComplexType(input)) {
        return op::DataType::DT_COMPLEX64;
    } else if (IsFloatingType(input)) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    bool is910BSocVersion =
        (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
         IsRegBase());
    const std::initializer_list<DataType> DTYPE_SUPPORT_LIST =
        is910BSocVersion ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;

    // 检查self的数据类型是否在addcdiv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    // 检查tensor1的数据类型是否在addcdiv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(tensor1, DTYPE_SUPPORT_LIST, return false);

    // 检查tensor2的数据类型是否在addcdiv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(tensor2, DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckPromoteType(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* out)
{
    // 检查tensor1和tensor2能否做数据类型推导
    op::DataType promoteType = op::PromoteType(tensor1->GetDataType(), tensor2->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "tensor1 dtype %s and tensor2 dtype %s can not promote dtype.",
            op::ToString(tensor1->GetDataType()).GetString(), op::ToString(tensor2->GetDataType()).GetString());
        return false;
    }

    // 检查self和tensor1 * tensor2的结果能否做数据类型推导
    promoteType = op::PromoteType(self->GetDataType(), promoteType);
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self dtype %s and tensor1 * tensor2 dtype %s * %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(tensor1->GetDataType()).GetString(),
            op::ToString(tensor2->GetDataType()).GetString());
        return false;
    }

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static bool CheckMaxDimension(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MAX_DIM(tensor1, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MAX_DIM(tensor2, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIM, return false);

    return true;
}

static bool CheckInAndOutShape(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* out)
{
    op::Shape shape1;
    op::Shape shape2;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(tensor1, tensor2, shape1, return false);
    OP_CHECK_BROADCAST_WITH_SHAPE(self, shape1, return false);
    BroadcastInferShape(self->GetViewShape(), shape1, shape2);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, shape2, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
    const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, tensor1, tensor2, value, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, tensor1, tensor2), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, tensor1, tensor2, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查最大维度是否超过8
    CHECK_RET(CheckMaxDimension(self, tensor1, tensor2, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查三输入是否能broadcast以及计算后的shape与out shape是否一致
    CHECK_RET(CheckInAndOutShape(self, tensor1, tensor2, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool IsMixedDType(const aclTensor* self, const aclScalar* value)
{
    auto valueDtype = GetScalarDefaultDtype(value->GetDataType());
    auto selfDtype = self->GetDataType();
    return (selfDtype == op::DataType::DT_FLOAT16 || selfDtype == op::DataType::DT_BF16) &&
           (valueDtype == op::DataType::DT_FLOAT);
}

aclnnStatus aclnnAddcdivGetWorkspaceSize(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
    const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnAddcdiv, DFX_IN(self, tensor1, tensor2, value), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, tensor1, tensor2, value, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || tensor1->IsEmpty() || tensor2->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 对输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType =
        op::PromoteType(self->GetDataType(), op::PromoteType(tensor1->GetDataType(), tensor2->GetDataType()));

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入tensor1转换成连续的tensor
    auto tensor1Contiguous = l0op::Contiguous(tensor1, uniqueExecutor.get());
    CHECK_RET(tensor1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto tensor1Casted = l0op::Cast(tensor1Contiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(tensor1Casted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入tensor2转换成连续的tensor
    auto tensor2Contiguous = l0op::Contiguous(tensor2, uniqueExecutor.get());
    CHECK_RET(tensor2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto tensor2Casted = l0op::Cast(tensor2Contiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(tensor2Casted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果是混合数据类型（self为bf16或float16,且value为float32），则value dtype保持float32类型
    bool isToFloat = IsRegBase() &&
                     IsMixedDType(self, value) && promoteType != op::DataType::DT_DOUBLE;
    auto valueDtype = isToFloat ? op::DataType::DT_FLOAT : promoteType;
    auto valueTensor = uniqueExecutor.get()->ConvertToTensor(value, valueDtype);
    CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行Addcdiv计算
    auto addcdivOpOut = l0op::Addcdiv(selfCasted, tensor1Casted, tensor2Casted, valueTensor, uniqueExecutor.get());
    CHECK_RET(addcdivOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(addcdivOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddcdiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddcdiv);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAddcdivGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnAddcdivGetWorkspaceSize(selfRef, tensor1, tensor2, value, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceAddcdiv(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceAddcdiv);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
