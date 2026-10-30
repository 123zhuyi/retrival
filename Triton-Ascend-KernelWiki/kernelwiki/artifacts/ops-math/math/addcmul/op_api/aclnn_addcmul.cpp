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
 * \file aclnn_addcmul.cpp
 * \brief
 */

#include "aclnn_addcmul.h"
#include "addcmul.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "math/axpy_v2/op_api/axpy_v2.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_SHAPE_LENGTH = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> PROMOTE_UN_SUPPORT_LIST = {
    op::DataType::DT_INT16, op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_AXPY_V2_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
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

static bool CheckNotNull(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(tensor1, return false);
    OP_CHECK_NULL(tensor2, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2)
{
    auto supportList = GetDtypeSupportList();

    // 检查self的数据类型是否在addcmul算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查tensor1的数据类型是否在addcmul算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(tensor1, supportList, return false);

    // 检查tensor2的数据类型是否在addcmul算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(tensor2, supportList, return false);
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

    if (CheckType(promoteType, PROMOTE_UN_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "addcmul can not support promote dtype %s, self dtype is %s and tensor1 * tensor2 dtype is %s * %s",
            op::ToString(promoteType).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(tensor1->GetDataType()).GetString(), op::ToString(tensor2->GetDataType()).GetString());
        return false;
    }

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* out)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(tensor1->GetStorageFormat()) ||
        op::IsPrivateFormat(tensor2->GetStorageFormat()) || op::IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, self [%s], tensor1 [%s], tensor2 [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(tensor1->GetStorageFormat()).GetString(),
            op::ToString(tensor2->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SHAPE_LENGTH, return false);
    OP_CHECK_MAX_DIM(tensor1, MAX_SHAPE_LENGTH, return false);
    OP_CHECK_MAX_DIM(tensor2, MAX_SHAPE_LENGTH, return false);

    op::Shape broadcastShape;
    if (!BroadcastInferShape(tensor1->GetViewShape(), tensor2->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of tensor1 and tensor2 can't broadcast.");
        return false;
    }

    if (!BroadcastInferShape(self->GetViewShape(), broadcastShape, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of self and other can't broadcast.");
        return false;
    }

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value, aclTensor* out)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(self, tensor1, tensor2, value, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(self, tensor1, tensor2), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查输入能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_COND(CheckPromoteType(self, tensor1, tensor2, out), ACLNN_ERR_PARAM_INVALID, "CheckPromoteType failed!");

    // 4. 检查数据格式是否支持
    CHECK_COND(CheckFormat(self, tensor1, tensor2, out), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed!");

    // 5. 检查输入是否能broadcast
    CHECK_COND(CheckShape(self, tensor1, tensor2, out), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    return ACLNN_SUCCESS;
}

static bool IsMixedDType(const aclTensor* self, const aclScalar* value)
{
    auto valueDtype = GetScalarDefaultDtype(value->GetDataType());
    auto selfDtype = self->GetDataType();
    return (selfDtype == op::DataType::DT_FLOAT16 || selfDtype == op::DataType::DT_BF16) &&
           (valueDtype == op::DataType::DT_FLOAT);
}

static inline bool IsEqualToOne(const aclScalar* value)
{
    if (IsComplexType(value->GetDataType())) {
        return false;
    }
    return !(value->ToFloat() > 1 || value->ToFloat() < 1);
}

static bool IsSupportAxpyV2(const DataType promoteType)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        return CheckType(promoteType, ASCEND910B_AXPY_V2_DTYPE_SUPPORT_LIST);
    }
    return false;
}

static bool CanXBroadcastToY(const aclTensor* tensorX, const aclTensor* tensorY)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(tensorX->GetViewShape(), tensorY->GetViewShape(), broadcastShape)) {
        return false;
    }
    if (broadcastShape != tensorY->GetViewShape()) {
        return false;
    }
    return true;
}

aclnnStatus aclnnAddcmulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnAddcmul, DFX_IN(self, tensor1, tensor2, value), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParams(self, tensor1, tensor2, value, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_COND(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR, "CREATE_EXECUTOR failed!");

    // 算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty() || tensor1->IsEmpty() || tensor2->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 对输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType =
        op::PromoteType(self->GetDataType(), op::PromoteType(tensor1->GetDataType(), tensor2->GetDataType()));

    auto self_contiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_COND(self_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "InitializeTensor self failed!");

    auto self_casted = l0op::Cast(self_contiguous, promoteType, uniqueExecutor.get());
    CHECK_COND(self_casted != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast self failed!");

    auto tensor1_contiguous = l0op::Contiguous(tensor1, uniqueExecutor.get());
    CHECK_COND(tensor1_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "InitializeTensor tensor1 failed!");

    auto tensor1_casted = l0op::Cast(tensor1_contiguous, promoteType, uniqueExecutor.get());
    CHECK_COND(tensor1_casted != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast tensor1 failed!");

    auto tensor2_contiguous = l0op::Contiguous(tensor2, uniqueExecutor.get());
    CHECK_COND(tensor2_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "InitializeTensor tensor2 failed!");

    auto tensor2_casted = l0op::Cast(tensor2_contiguous, promoteType, uniqueExecutor.get());
    CHECK_COND(tensor2_casted != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast tensor2 failed!");

    // 如果是混合数据类型（self为bf16或float16,且value为float32），则value dtype保持float32类型
    bool isToFloat = IsRegBase() &&
                     IsMixedDType(self, value) && promoteType != op::DataType::DT_DOUBLE;
    auto valueDtype = isToFloat ? op::DataType::DT_FLOAT : promoteType;
    auto valueTensor = uniqueExecutor.get()->ConvertToTensor(value, valueDtype);

    // 调用Addcmul算子kernel
    const aclTensor* addcmul_op_out = nullptr;
    if (IsEqualToOne(value) && IsSupportAxpyV2(promoteType) && CanXBroadcastToY(tensor2_casted, self_casted)) {
        addcmul_op_out = l0op::AxpyV2(self_casted, tensor1_casted, tensor2_casted, uniqueExecutor.get());
    } else {
        addcmul_op_out = l0op::Addcmul(self_casted, tensor1_casted, tensor2_casted, valueTensor, uniqueExecutor.get());
    }
    CHECK_COND(addcmul_op_out != nullptr, ACLNN_ERR_INNER_NULLPTR, "Addcmul failed!");

    // 固定写法，将计算结果转换成输出out的数据类型
    auto cast_out = l0op::Cast(addcmul_op_out, out->GetDataType(), uniqueExecutor.get());
    CHECK_COND(cast_out != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast out failed!");

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto view_copy_result = l0op::ViewCopy(cast_out, out, uniqueExecutor.get());
    CHECK_COND(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR, "viewcopy out failed!");

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceAddcmulGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnAddcmulGetWorkspaceSize(selfRef, tensor1, tensor2, value, out, workspaceSize, executor);
}

aclnnStatus aclnnAddcmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddcmul);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAddcmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceAddcmul);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
