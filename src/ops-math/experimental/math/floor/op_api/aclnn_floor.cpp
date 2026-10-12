/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_floor.h"
#include "floor.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "op_api/level2_base.h"
#include "op_api/aclnn_check.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910 = {
    DataType::DT_INT64,  DataType::DT_INT32,  DataType::DT_INT16,  DataType::DT_INT8,
    DataType::DT_UINT64, DataType::DT_UINT32, DataType::DT_UINT16, DataType::DT_UINT8,
    DataType::DT_DOUBLE, DataType::DT_FLOAT,  DataType::DT_FLOAT16};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910B = {
    DataType::DT_INT64,  DataType::DT_INT32,  DataType::DT_INT16,   DataType::DT_INT8,
    DataType::DT_UINT64, DataType::DT_UINT32, DataType::DT_UINT16,  DataType::DT_UINT8,
    DataType::DT_DOUBLE, DataType::DT_FLOAT,  DataType::DT_FLOAT16, DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // self和out数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool is910BSocVersion = npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch);
    const std::initializer_list<DataType> DTYPE_SUPPORT_LIST =
        is910BSocVersion ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, self [%s], out [%s]",
            ToString(self->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParamsFloor(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckSameShape1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsFloor(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* floorResult = nullptr;
    if (IsIntegralType(self->GetDataType())) {
        floorResult = selfContiguous;
    } else {
        // 调用l0算子Floor进行计算
        floorResult = l0op::Floor(selfContiguous, uniqueExecutor.get());
    }
    CHECK_RET(floorResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(floorResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFloorGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFloor, DFX_IN(self), DFX_OUT(out));
    return GetWorkspaceSizeCommon(self, out, workspaceSize, executor);
}

aclnnStatus aclnnFloor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFloor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceFloorGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFloor, DFX_IN(selfRef), DFX_OUT(selfRef));
    return GetWorkspaceSizeCommon(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceFloor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFloor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}