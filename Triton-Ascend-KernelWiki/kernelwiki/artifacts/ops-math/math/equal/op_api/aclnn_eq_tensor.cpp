/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_eq_tensor.h"
#include "equal.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "op_api/data_type_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

/* Equal 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             Equal(workspace_4)
 *                    |
 *              Cast(workspace_5)
 *                    |
 *                 ViewCopy
 *                    |
 *                  result
 */

// 根据API定义，需要列出所能支持的所有dtype (1980)
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,  op::DataType::DT_UINT8,  op::DataType::DT_UINT64,
    op::DataType::DT_UINT32,    op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

// 根据API定义，需要列出所能支持的所有dtype (1971)
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,   op::DataType::DT_UINT8,     op::DataType::DT_BF16,
    op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

// 根据API定义，需要列出所能支持的所有dtype (1982)
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,   op::DataType::DT_UINT8,     op::DataType::DT_BF16,
    op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128,
    op::DataType::DT_UINT64};

// 列出output所支持的dtype
// 除910B-950以外版本,
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,  op::DataType::DT_UINT8,  op::DataType::DT_UINT64,
    op::DataType::DT_UINT32,    op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

// 910B-910_93
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,   op::DataType::DT_UINT8,     op::DataType::DT_BF16,
    op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

// 950版本支持
static const std::initializer_list<op::DataType> REGBASE_OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,       op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,     op::DataType::DT_BOOL,  op::DataType::DT_UINT16,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

// 算子支持的最大维度
static const size_t DIM_SUPPORT_MAX = 8;

static const std::initializer_list<op::DataType> GetInputDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201: {
            return DTYPE_SUPPORT_910B_LIST;
        }
        case NpuArch::DAV_3510: {
            return REGBASE_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return DTYPE_SUPPORT_910_LIST;
        }
        default: {
            return DTYPE_SUPPORT_910_LIST;
        }
    }
}

static inline const std::initializer_list<op::DataType>& GetOutputDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        return OUT_DTYPE_SUPPORT_910B_LIST;
    } else if (IsRegBase(npuArch)) {
        return REGBASE_OUT_DTYPE_SUPPORT_LIST;
    }
    return OUT_DTYPE_SUPPORT_910_LIST;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other, const aclTensor* out, DataType& promoteType)
{
    const std::initializer_list<op::DataType> inputSupportList = GetInputDtypeSupportList();
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        // 检查self的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(self, inputSupportList, return false);

        // 检查other的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(other, inputSupportList, return false);
        auto outputSupportList = GetOutputDtypeSupportList();
        OP_CHECK_DTYPE_NOT_SUPPORT(out, outputSupportList, return false);
    } else {
        // 检查out的数据类型是否在equal算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(out, inputSupportList, return false);
    }

    // 如果soc是1980芯片，则不支持DT_BF16，需要校验拦截，否则Cast报错
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
        if (self->GetDataType() == op::DataType::DT_BF16 || other->GetDataType() == op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The current soc version does not support the DT_BF16 data type.");
            return false;
        }
    }

    // 检查self和other能否做数据类型推导
    if (IsRegBase(npuArch)) {
        promoteType = op::BinaryOpTypePromote(self, other);
    } else {
        promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    }
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
        return false;
    }

    // 检查promoteType的数据类型是否在equal算子的支持列表内
    if (!CheckType(promoteType, inputSupportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Self dtype %s and other dtype %s get promoteType dtype %s should be in "
            "dtype support list [%s].",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString(),
            op::ToString(promoteType).GetString(), op::ToString(inputSupportList).GetString());
        return false;
    }

    // 检查BOOL类型能否转换为输出的数据类型(算子返回的都是BOOL类型)
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(DataType::DT_BOOL, out->GetDataType(), return false);

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, DIM_SUPPORT_MAX, return false);
    OP_CHECK_MAX_DIM(other, DIM_SUPPORT_MAX, return false);
    OP_CHECK_MAX_DIM(out, DIM_SUPPORT_MAX, return false);

    op::Shape outShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, outShape, return false);

    if (outShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "BroadcastShape %s is not equal out's shape %s.",
            op::ToString(outShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* other, const aclTensor* out, DataType& promoteType)
{
    // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other, out, promoteType), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查双输入是否能broadcast,检查boradcast后的输出与out是否一致
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEqTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnEqTensor, DFX_IN(self, other), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 1. 检查三个入参参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，参数检查
    DataType promoteType = DataType::DT_UNDEFINED;
    auto ret = CheckParams(self, other, out, promoteType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果两个入参其中有一个为空，则返回空
    if (self->IsEmpty() || other->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Equal算子kernel
    const aclTensor* equalOpOut = l0op::Equal(selfCasted, otherCasted, uniqueExecutor.get());
    CHECK_RET(equalOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果(BOOL)转换成输出out的数据类型
    auto castOut = l0op::Cast(equalOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckInplace(const aclTensor* selfRef, const aclTensor* other)
{
    OP_CHECK_NULL(selfRef, return false);
    OP_CHECK_NULL(other, return false);

    op::Shape broadcastShape;
    OP_CHECK(
        BroadcastInferShape(selfRef->GetViewShape(), other->GetViewShape(), broadcastShape),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of selfRef and other can't broadcast, got %s, %s.",
            op::ToString(selfRef->GetViewShape()).GetString(), op::ToString(other->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        selfRef->GetViewShape() == broadcastShape,
        OP_LOGE(
            ACLNN_ERR_PARAM_NULLPTR, "Expected shape of selfRef should be %s, but got %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(selfRef->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceEqTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckInplace(selfRef, other);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnEqTensorGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

aclnnStatus aclnnEqTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnEqTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceEqTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceEqTensor);
    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
