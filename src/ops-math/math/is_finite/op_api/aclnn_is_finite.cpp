/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_isfinite.h"
#include "is_finite.h"
#include "conversion/fill/op_api/fill.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace {
constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT32, DataType::DT_INT64,
    DataType::DT_INT16, DataType::DT_INT8,    DataType::DT_UINT8,  DataType::DT_BOOL};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910B = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_DOUBLE, DataType::DT_INT32,
    DataType::DT_INT64, DataType::DT_INT16,   DataType::DT_INT8, DataType::DT_UINT8,  DataType::DT_BOOL};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool is910BSocVersion =
        (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch));
    const std::initializer_list<DataType> DTYPE_SUPPORT_LIST =
        is910BSocVersion ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    // 检查out的数据类型是否为BOOL
    OP_CHECK_DTYPE_NOT_MATCH(out, DataType::DT_BOOL, return false);

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

    // self与out的shape需一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetTrueTensor(const aclTensor* self, aclOpExecutor* executor)
{
    aclScalar* scalar = executor->AllocScalar(1);
    CHECK_RET(scalar != nullptr, nullptr);
    auto valueTensor = executor->ConvertToTensor(scalar, op::DataType::DT_BOOL);
    CHECK_RET(valueTensor != nullptr, nullptr);
    auto outputDims = op::ToShapeVector(self->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(outputDims.data(), outputDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    CHECK_RET(dimTensor != nullptr, nullptr);
    auto trueTensor = l0op::Fill(dimTensor, valueTensor, dimArray, executor);
    return trueTensor;
}
} // namespace

aclnnStatus aclnnIsFiniteGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnIsFinite, DFX_IN(self), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* isfiniteResult = nullptr;

    if (!IsFloatingType(self->GetDataType())) {
        // 场景一：非浮点数调用Fill
        isfiniteResult = GetTrueTensor(self, uniqueExecutor.get());
    } else {
        // 场景二：直接调l0op::IsFinite
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        isfiniteResult = l0op::IsFinite(selfContiguous, uniqueExecutor.get());
    }
    CHECK_RET(isfiniteResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(isfiniteResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIsFinite(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIsFinite);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
