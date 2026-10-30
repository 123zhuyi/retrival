/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_add_mat_mat_elements.cpp
 * @brief ACLNN L2 API 实现 - AddMatMatElements 算子
 *
 * 计算公式：c_out[i] = c[i] × beta + alpha × a[i] × b[i]
 *
 * 两段式流程：
 *   1. GetWorkspaceSize: 参数检查 → Contiguous → L0 调用 → ViewCopy → 获取 workspace 大小
 *   2. Execute: 调用 CommonOpExecutorRun
 */

#include "aclnn_add_mat_mat_elements.h"
#include "add_mat_mat_elements.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

// 支持的 dtype 列表（Ascend950 / DAV_3510）
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsDtypeSupported(DataType dtype) { return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST); }

static bool CheckNotNull(
    const aclTensor* a, const aclTensor* b, const aclTensor* c, const aclScalar* alpha, const aclScalar* beta,
    const aclTensor* cOut)
{
    OP_CHECK_NULL(a, return false);
    OP_CHECK_NULL(b, return false);
    OP_CHECK_NULL(c, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(beta, return false);
    OP_CHECK_NULL(cOut, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* a, const aclTensor* b, const aclTensor* c, const aclTensor* cOut)
{
    // 所有 tensor 的 dtype 必须一致
    OP_CHECK_DTYPE_NOT_MATCH(a, b->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(a, c->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(a, cOut->GetDataType(), return false);

    if (!IsDtypeSupported(a->GetDataType())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "AddMatMatElements: unsupported dtype=%d. "
            "Supported: FLOAT16, FLOAT, BF16.",
            static_cast<int>(a->GetDataType()));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* a, const aclTensor* b, const aclTensor* c, const aclTensor* cOut)
{
    if (IsPrivateFormat(a->GetStorageFormat()) || IsPrivateFormat(b->GetStorageFormat()) ||
        IsPrivateFormat(c->GetStorageFormat()) || IsPrivateFormat(cOut->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AddMatMatElements: private format not supported.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* a, const aclTensor* b, const aclTensor* c, const aclTensor* cOut)
{
    OP_CHECK_MAX_DIM(a, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(b, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(c, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(cOut, ACLNN_MAX_SHAPE_RANK, return false);

    // ISSUE-005 修复：校验 a/b/c/cOut 四个 Tensor 的 shape 一致性
    OP_CHECK_SHAPE_NOT_EQUAL(a, b, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(a, c, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(a, cOut, return false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* a, const aclTensor* b, const aclTensor* c, const aclScalar* alpha, const aclScalar* beta,
    const aclTensor* cOut)
{
    if (!CheckNotNull(a, b, c, alpha, beta, cOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "AddMatMatElements: CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(a, b, c, cOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AddMatMatElements: CheckDtypeValid failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(a, b, c, cOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AddMatMatElements: CheckFormat failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(a, b, c, cOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AddMatMatElements: CheckShape failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 */
extern "C" aclnnStatus aclnnAddMatMatElementsGetWorkspaceSize(
    const aclTensor* a, const aclTensor* b, const aclTensor* c, const aclScalar* alpha, const aclScalar* beta,
    aclTensor* cOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddMatMatElements, DFX_IN(a, b, c), DFX_OUT(cOut));

    // ISSUE-003 修复：输出指针解引用前必须判空
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "AddMatMatElements: workspaceSize or executor is nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(a, b, c, alpha, beta, cOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空 Tensor 快速返回
    if (a->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 保证输入连续
    auto aContiguous = l0op::Contiguous(a, uniqueExecutor.get());
    CHECK_RET(aContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto bContiguous = l0op::Contiguous(b, uniqueExecutor.get());
    CHECK_RET(bContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto cContiguous = l0op::Contiguous(c, uniqueExecutor.get());
    CHECK_RET(cContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用 L0 API
    const aclTensor* opResult = l0op::AddMatMatElements(
        aContiguous, bContiguous, cContiguous, alpha->ToFloat(), beta->ToFloat(), uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出非连续处理
    auto viewCopyResult = l0op::ViewCopy(opResult, cOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 */
extern "C" aclnnStatus aclnnAddMatMatElements(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddMatMatElements);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
