/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_hardtanh.h"
#include "clip_by_value.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64,
    DataType::DT_INT16,   DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_INT32,  DataType::DT_INT64, DataType::DT_INT16,
    DataType::DT_INT8,    DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out, const aclScalar* min, const aclScalar* max)
{
    const auto& supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(min, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(max, supportList, return false);

    // self和out数据类型必须一样
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

static bool CheckMinMaxValue(const aclScalar* min, const aclScalar* max)
{
    const auto& supportList = GetDtypeSupportList();
    // 检查min的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(min, supportList, return false);

    if (IsIntegralType(min->GetDataType())) {
        int64_t imin = min->ToInt64();
        int64_t imax = max->ToInt64();
        if (imin > imax) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "min must less than max or equal to max, but min is %ld and max is %ld.", imin,
                imax);
            return false;
        }
    } else if (IsFloatingType(min->GetDataType())) {
        double fmin = min->ToDouble();
        double fmax = max->ToDouble();
        if (fmin > fmax) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "min must less than max or equal to max, but min is %f and max is %f.", fmin,
                fmax);
            return false;
        }
    }

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    return true;
}

static void CheckFormat(const aclTensor* self)
{
    // 检查format，若是NZ格式，则添加警告
    if (self->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("Format of self gets [%s], this format may lead to precision failure.", 
        op::ToString(self->GetStorageFormat()).GetString());
    }
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, const aclTensor* out)
{
    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out, clipValueMin, clipValueMax), ACLNN_ERR_PARAM_INVALID);

    // 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查min是否小于max
    CHECK_RET(CheckMinMaxValue(clipValueMin, clipValueMax), ACLNN_ERR_PARAM_INVALID);

    // 检查format，若是NZ格式，则添加警告
    CheckFormat(self);

    return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_NOT_NULL(self, clipValueMin, clipValueMax, out);
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, clipValueMin, clipValueMax, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // scalar 转 tensor
    auto minTensor = uniqueExecutor.get()->ConvertToTensor(clipValueMin, self->GetDataType());
    auto maxTensor = uniqueExecutor.get()->ConvertToTensor(clipValueMax, self->GetDataType());

    // 调用l0算子ClipByValueV2进行计算
    auto hardtanhResult = l0op::ClipByValueV2(selfContiguous, minTensor, maxTensor, uniqueExecutor.get());
    CHECK_RET(hardtanhResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(hardtanhResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardtanhGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnHardtanh, DFX_IN(self, clipValueMin, clipValueMax), DFX_OUT(out));
    return GetWorkspaceSizeCommon(self, clipValueMin, clipValueMax, out, workspaceSize, executor);
}

aclnnStatus aclnnHardtanh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHardtanh);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceHardtanhGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* clipValueMin, const aclScalar* clipValueMax, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceHardtanh, DFX_IN(selfRef, clipValueMin, clipValueMax), DFX_OUT(selfRef));
    return GetWorkspaceSizeCommon(selfRef, clipValueMin, clipValueMax, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceHardtanh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceHardtanh);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
