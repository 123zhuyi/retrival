/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ge_tensor.h"
#include "greater_equal.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,  DataType::DT_INT32,  DataType::DT_FLOAT16, DataType::DT_INT8,
    DataType::DT_DOUBLE, DataType::DT_INT16,  DataType::DT_INT64,   DataType::DT_UINT64,
    DataType::DT_UINT32, DataType::DT_UINT16, DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,  DataType::DT_INT32,  DataType::DT_FLOAT16, DataType::DT_INT8,
    DataType::DT_DOUBLE, DataType::DT_INT16,  DataType::DT_INT64,   DataType::DT_UINT64,
    DataType::DT_UINT32, DataType::DT_UINT16, DataType::DT_UINT8,   DataType::DT_BF16};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32,      DataType::DT_FLOAT16,   DataType::DT_INT8,   DataType::DT_DOUBLE,
    DataType::DT_INT16, DataType::DT_INT64,      DataType::DT_UINT64,    DataType::DT_UINT32, DataType::DT_UINT16,
    DataType::DT_UINT8, DataType::DT_COMPLEX128, DataType::DT_COMPLEX64, DataType::DT_BOOL,   DataType::DT_BF16};

static const size_t DIM_BOUND = 8;

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckSocExtraType(const DataType dtype)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (dtype == op::DataType::DT_BF16 && (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch))) {
        return true;
    }
    return false;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* out)
{
    // 检查out的数据类型是否在支持列表内，当前CanCast能力与cast算子不一致，需要在这里判断
    if (!CheckType(out->GetDataType(), OUT_DTYPE_SUPPORT_LIST) && !CheckSocExtraType(out->GetDataType())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "aclnnGeTensor not implemented for out dtype %s.",
            op::ToString(out->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckPromoteType(const aclTensor* self, const aclTensor* other, const aclTensor* out, DataType& promoteType)
{
    const auto& supportList = GetDtypeSupportList();
    // 类型提升判断，将提升后的数据类型返回
    promoteType = PromoteType(self->GetDataType(), other->GetDataType());
    // 如果有经过评审可以进行的转换，在这里添加并修改promoteType的值
    if (promoteType == DataType::DT_BOOL) {
        promoteType = DataType::DT_INT8;
    }

    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            ToString(self->GetDataType()).GetString(), ToString(other->GetDataType()).GetString());
        return false;
    }

    if (!CheckType(promoteType, supportList) && !CheckSocExtraType(promoteType)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "aclnnGeTensor not implemented for input dtype %s.",
            ToString(promoteType).GetString());
        return false;
    }
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), promoteType, return false);
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), promoteType, return false);
    }
    // 检查计算结果的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(DataType::DT_BOOL, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, DIM_BOUND, return false);
    OP_CHECK_MAX_DIM(other, DIM_BOUND, return false);
    OP_CHECK_MAX_DIM(out, DIM_BOUND, return false);

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
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和other的数据类型能否提升
    CHECK_RET(CheckPromoteType(self, other, out, promoteType), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnGeTensorCommon(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    DataType promoteType;
    auto ret = CheckParams(self, other, out, promoteType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理，因为已经通过了broadcast判断，只要有一个tensor为空，另一个tensor经过broadcast后一定也为空，out的resize也已经在外部处理，直接返回
    if (self->IsEmpty() || other->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 处理self输入
    const aclTensor* selfProcessed = nullptr;
    if (promoteType == self->GetDataType() && l0op::IsGreaterEqualSupportNonContiguous(self)) {
        selfProcessed = uniqueExecutor.get()->CreateView(
            self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
    } else {
        // self如果非连续，需要转连续
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入self的数据类型转换成推导后的数据类型
        selfProcessed = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    }
    CHECK_RET(selfProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 处理other输入
    const aclTensor* otherProcessed = nullptr;
    if (promoteType == other->GetDataType() && l0op::IsGreaterEqualSupportNonContiguous(other)) {
        otherProcessed = uniqueExecutor.get()->CreateView(
            other, other->GetViewShape(), other->GetStorageShape(), other->GetViewStrides(), other->GetViewOffset());
    } else {
        // other如果非连续，需要转连续
        auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入other的数据类型转换成推导后的数据类型
        otherProcessed = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    }
    CHECK_RET(otherProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子GreaterEqual进行计算
    auto greaterEqualResult = l0op::GreaterEqual(selfProcessed, otherProcessed, uniqueExecutor.get());
    CHECK_RET(greaterEqualResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输出的数据类型转换成推导后的数据类型
    auto greaterEqualResultCasted = l0op::Cast(greaterEqualResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(greaterEqualResultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(greaterEqualResultCasted, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGeTensor, DFX_IN(self, other), DFX_OUT(out));
    return aclnnGeTensorCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnGeTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGeTensor);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceGeTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceGeTensor, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    return aclnnGeTensorCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceGeTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceGeTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
