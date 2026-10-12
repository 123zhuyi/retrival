/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gcd.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "gcd.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
static constexpr int64_t GCD_SMALL_MIXED_FUSED_MAX_ELEMENTS = 4096;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> GCD_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_INT32, op::DataType::DT_INT64,
};

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    const auto selfType = self->GetDataType();
    const auto otherType = other->GetDataType();
    const auto outType = out->GetDataType();
    if (!CheckType(selfType, GCD_DTYPE_SUPPORT_LIST) || !CheckType(otherType, GCD_DTYPE_SUPPORT_LIST) ||
        !CheckType(outType, GCD_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd dtypes %s, %s and %s must be one of %s.",
                op::ToString(selfType).GetString(), op::ToString(otherType).GetString(),
                op::ToString(outType).GetString(), op::ToString(GCD_DTYPE_SUPPORT_LIST).GetString());
        return false;
    }

    op::DataType promoteType = op::PromoteType(selfType, otherType);
    if (promoteType == DataType::DT_UNDEFINED || !CheckType(promoteType, GCD_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Self dtype %s and other dtype %s get promoteType dtype %s should be in dtype support list %s.",
                op::ToString(selfType).GetString(), op::ToString(otherType).GetString(),
                op::ToString(promoteType).GetString(), op::ToString(GCD_DTYPE_SUPPORT_LIST).GetString());
        return false;
    }
    return true;
}

// 检查self、other、out的shape是否满足broadcast规则
static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // self和other的维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(other, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    // self和other能否做broadcast
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);

    // broadcast后的shape需要与out一致。
    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected out tensor shape to be %s, but got %s.",
                op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输出shape
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool IsSmallMixedFusedToOutput(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    const auto selfType = self->GetDataType();
    const auto otherType = other->GetDataType();
    const auto outType = out->GetDataType();
    if (!op::IsContiguous(out)) {
        return false;
    }
    if (out->Numel() <= 0 || out->Numel() > GCD_SMALL_MIXED_FUSED_MAX_ELEMENTS) {
        return false;
    }

    return gcd::IsRegisteredMixedKernelSignature(selfType, otherType, outType);
}

static aclnnStatus ReleaseExecutor(UniqueExecutor& uniqueExecutor, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static aclnnStatus BuildSmallMixedPath(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                       UniqueExecutor& uniqueExecutor, uint64_t* workspaceSize,
                                       aclOpExecutor** executor, bool& finished)
{
    finished = false;
    if (!IsSmallMixedFusedToOutput(self, other, out)) {
        return ACLNN_SUCCESS;
    }

    auto mixedOut = l0op::GcdWithOutputType(self, other, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(mixedOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(mixedOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    finished = true;
    return ReleaseExecutor(uniqueExecutor, workspaceSize, executor);
}

static aclnnStatus BuildPromotedInputs(const aclTensor* self, const aclTensor* other, const aclTensor*& selfCasted,
                                       const aclTensor*& otherCasted, UniqueExecutor& uniqueExecutor)
{
    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    if (self->GetDataType() != promoteType) {
        selfCasted = l0op::Cast(self, promoteType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (other->GetDataType() != promoteType) {
        otherCasted = l0op::Cast(other, promoteType, uniqueExecutor.get());
        CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static bool CheckRegisteredSameDtypeKernelInputs(const aclTensor* self, const aclTensor* other)
{
    const auto selfType = self->GetDataType();
    const auto otherType = other->GetDataType();
    if (!CheckType(selfType, GCD_DTYPE_SUPPORT_LIST) || !CheckType(otherType, GCD_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd kernel input dtypes %s and %s must be one of %s.",
                op::ToString(selfType).GetString(), op::ToString(otherType).GetString(),
                op::ToString(GCD_DTYPE_SUPPORT_LIST).GetString());
        return false;
    }
    if (selfType != otherType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd kernel requires same-dtype inputs, but got self %s and other %s.",
                op::ToString(selfType).GetString(), op::ToString(otherType).GetString());
        return false;
    }
    return true;
}

static aclnnStatus BuildDirectOutputPath(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                         UniqueExecutor& uniqueExecutor, uint64_t* workspaceSize,
                                         aclOpExecutor** executor, bool& finished)
{
    finished = false;
    if (self->GetDataType() != out->GetDataType() || other->GetDataType() != out->GetDataType() ||
        !op::IsContiguous(out)) {
        return ACLNN_SUCCESS;
    }

    auto directOut = l0op::GcdToOutput(self, other, out, uniqueExecutor.get());
    CHECK_RET(directOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    finished = true;
    return ReleaseExecutor(uniqueExecutor, workspaceSize, executor);
}

static aclnnStatus BuildGeneralOutputPath(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                          UniqueExecutor& uniqueExecutor)
{
    auto gcdOut = l0op::Gcd(self, other, uniqueExecutor.get());
    CHECK_RET(gcdOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = gcdOut;
    if (gcdOut->GetDataType() != out->GetDataType()) {
        castOut = l0op::Cast(gcdOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGcdGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                     uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGcd, DFX_IN(self, other), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || other->IsEmpty()) {
        return ReleaseExecutor(uniqueExecutor, workspaceSize, executor);
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool finished = false;
    ret = BuildSmallMixedPath(selfContiguous, otherContiguous, out, uniqueExecutor, workspaceSize, executor, finished);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (finished) {
        return ACLNN_SUCCESS;
    }

    auto selfCasted = selfContiguous;
    auto otherCasted = otherContiguous;
    ret = BuildPromotedInputs(selfContiguous, otherContiguous, selfCasted, otherCasted, uniqueExecutor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckRegisteredSameDtypeKernelInputs(selfCasted, otherCasted), ACLNN_ERR_PARAM_INVALID);

    ret = BuildDirectOutputPath(selfCasted, otherCasted, out, uniqueExecutor, workspaceSize, executor, finished);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (finished) {
        return ACLNN_SUCCESS;
    }

    ret = BuildGeneralOutputPath(selfCasted, otherCasted, out, uniqueExecutor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ReleaseExecutor(uniqueExecutor, workspaceSize, executor);
}

aclnnStatus aclnnGcd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGcd);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
