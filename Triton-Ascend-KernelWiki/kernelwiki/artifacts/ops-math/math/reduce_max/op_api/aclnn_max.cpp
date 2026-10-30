/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max.h"
#include "reduce_max.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM = 8;

static const std::initializer_list<DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_DOUBLE,  op::DataType::DT_BOOL};

static const std::initializer_list<DataType> DTYPE_SUPPORT_GE910B_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

// 判断芯片类型是否大于等于910B
static inline bool CheckSocVersionGe910B(void)
{
    return (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
            GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) ||
           IsRegBase();
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 获取芯片类型,判断是1971还是1980
    bool is910BSocVersion = CheckSocVersionGe910B();
    const std::initializer_list<DataType> CURRENT_DTYPE_SUPPORT_LIST =
        is910BSocVersion ? DTYPE_SUPPORT_GE910B_LIST : DTYPE_SUPPORT_910_LIST;

    OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, CURRENT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclIntArray* GetMaxDimListForTensor(const aclTensor* self, aclOpExecutor* executor)
{
    uint64_t dimNum = self->GetViewShape().GetDimNum();
    int64_t dimList[dimNum];
    for (size_t i = 0; i < dimNum; i++) {
        dimList[i] = i;
    }
    aclIntArray* dim = executor->AllocIntArray(dimList, dimNum);
    return dim;
}

aclnnStatus aclnnMaxGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMax, DFX_IN(self), DFX_OUT(out));

    // 参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto dim = GetMaxDimListForTensor(self, uniqueExecutor.get());

    // 空Tensor处理
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    DataType selfCastType =
        (self->GetDataType() == op::DataType::DT_BOOL) ? op::DataType::DT_FLOAT : self->GetDataType();
    auto selfCast = l0op::Cast(selfContiguous, selfCastType, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子Max进行计算
    auto maxResult = l0op::ReduceMax(selfCast, dim, false, true, uniqueExecutor.get());
    CHECK_RET(maxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckShapeAndScalarSame(maxResult, out), ACLNN_ERR_PARAM_INVALID);

    auto castMaxOut = l0op::Cast(maxResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(castMaxOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMax);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif