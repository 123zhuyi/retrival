/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_gt_tensor.h"
#include "greater.h"
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

/* Greater 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             Greater(workspace_4)
 *                    |
 *              Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

// 根据API定义，需要列出所能支持的所有dtype 910
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,  op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,   op::DataType::DT_UINT8,  op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_UINT16, op::DataType::DT_BOOL};

// out 910
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,     op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,      op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,    op::DataType::DT_BOOL,  op::DataType::DT_UINT16,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

// 根据API定义，需要列出所能支持的所有dtype 910B
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,  op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,   op::DataType::DT_UINT8,  op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_UINT16, op::DataType::DT_BF16,
    op::DataType::DT_BOOL};

// out 910B
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,       op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,     op::DataType::DT_BOOL,  op::DataType::DT_UINT16,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

// 算子支持的最大维度
static const size_t DIM_SUPPORT_MAX = 8;

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other, const aclTensor* out, DataType& promoteType)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool is910bSocVersion = (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch));
    const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST =
        is910bSocVersion ? DTYPE_SUPPORT_910B_LIST : DTYPE_SUPPORT_910_LIST;
    const std::initializer_list<op::DataType> CURRENT_OUT_DTYPE_SUPPORT_LIST =
        is910bSocVersion ? OUT_DTYPE_SUPPORT_910B_LIST : OUT_DTYPE_SUPPORT_910_LIST;

    // 检查out的数据类型是否在Greater算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, CURRENT_OUT_DTYPE_SUPPORT_LIST, return false);

    // 如果soc是1980芯片，则不支持DT_BF16，需要校验拦截，否则Cast报错
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
        if (self->GetDataType() == op::DataType::DT_BF16 || other->GetDataType() == op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The current soc version does not support the DT_BF16 data type.");
            return false;
        }
    }

    // 检查self和other能否做数据类型推导
    promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    if (promoteType == DataType::DT_BOOL) {
        promoteType = DataType::DT_INT8;
    }

    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
        return false;
    }

    // 检查promoteType的数据类型是否在Greater算子的支持列表内
    if (!CheckType(promoteType, CURRENT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Self dtype %s and other dtype %s get promoteType dtype %s should be in "
            "dtype support list [%s].",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString(),
            op::ToString(promoteType).GetString(), op::ToString(CURRENT_DTYPE_SUPPORT_LIST).GetString());
        return false;
    }
    if (IsRegBase(npuArch)) {
        // 检查self的数据类型是否合规
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), promoteType, return false);
        // 检查other的数据类型是否合规
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), promoteType, return false);
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

static aclnnStatus CheckParams4Gt(
    const aclTensor* self, const aclTensor* other, const aclTensor* out, DataType& promoteType)
{
    // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other, out, promoteType), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查双输入是否能broadcast,检查boradcast后的输出与out是否一致
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGtTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGtTensor, DFX_IN(self, other), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 1. 检查三个入参参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // 固定写法，参数检查
    DataType promoteType;
    auto result = CheckParams4Gt(self, other, out, promoteType);
    CHECK_RET(result == ACLNN_SUCCESS, result);

    // 两个入参Tensor其中有一个为空tensor时返回空
    if (other->IsEmpty() || self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous4Gt = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous4Gt != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous4Gt, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Greater算子kernel
    auto gtOpOut = l0op::Greater(selfCasted, otherCasted, uniqueExecutor.get());
    CHECK_RET(gtOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果(BOOL)转换成输出out的数据类型
    auto castOut = l0op::Cast(gtOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGtTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGtTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceGtTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnGtTensorGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceGtTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceGtTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
