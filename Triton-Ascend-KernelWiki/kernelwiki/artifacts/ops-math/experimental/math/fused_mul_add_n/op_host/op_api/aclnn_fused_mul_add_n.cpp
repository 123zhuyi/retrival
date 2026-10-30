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
 * @file aclnn_fused_mul_add_n.cpp
 * @brief ACLNN L2 API 实现 - FusedMulAddN (y = x1 * x3[0] + x2)
 *
 * ACLNN 接口采用两段式设计：
 * 1. aclnnFusedMulAddNGetWorkspaceSize - 参数检查、Contiguous、调度 L0、ViewCopy、计算 workspace 大小
 * 2. aclnnFusedMulAddN                 - 执行计算
 *
 * 文件组织：
 * - aclnn_fused_mul_add_n.h/cpp -> L2 API（本文件）：参数检查、Contiguous/ViewCopy 处理
 * - fused_mul_add_n.h/cpp       -> L0 API（底层实现）：形状推导、Kernel 调度
 *
 * 约束（真值，与 op_host tiling / op_graph proto 一致）：
 * - x1/x2/x3/y dtype 必须完全一致，且在支持集 {FLOAT, FLOAT16, BFLOAT16, INT32, INT16} 内；
 * - x1/x2/y shape 必须一致（逐元素，非矩阵乘）；
 * - x3 必须为单元素标量张量（ShapeSize == 1），仅取 x3[0]；
 * - 数据格式支持 ND（不支持私有格式）。
 */

#include "aclnn_fused_mul_add_n.h"
#include "fused_mul_add_n.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

// FusedMulAddN 支持的 dtype（x1/x2/x3/y 同 dtype）。与 op_host def / op_graph proto 一致。
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32,
    op::DataType::DT_INT16};

static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, const aclTensor* y)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x3, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, const aclTensor* y)
{
    // x1 dtype 必须在支持集内。
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    // x2/x3/y dtype 必须与 x1 完全一致。
    OP_CHECK_DTYPE_NOT_SAME(x2, x1, return false);
    OP_CHECK_DTYPE_NOT_SAME(x3, x1, return false);
    OP_CHECK_DTYPE_NOT_SAME(y, x1, return false);
    return true;
}

static bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(x1, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(x2, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(x3, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(y, MAX_DIM_LEN, return false);

    // x1/x2/y 逐元素，shape 必须一致（非矩阵乘、无广播）。
    OP_CHECK_SHAPE_NOT_EQUAL(x2, x1, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(y, x1, return false);

    // x3 必须为单元素标量张量（ShapeSize == 1），形态 [1]/[1,1] 等价，仅取 x3[0]。
    int64_t x3ShapeSize = x3->GetViewShape().GetShapeSize();
    OP_CHECK(
        x3ShapeSize == 1,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "x3 must be a single-element scalar tensor (ShapeSize == 1), but got ShapeSize %ld.", x3ShapeSize),
        return false);
    return true;
}

static bool CheckFormat(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, const aclTensor* y)
{
    OP_CHECK(
        !(IsPrivateFormat(x1->GetStorageFormat()) || IsPrivateFormat(x2->GetStorageFormat()) ||
          IsPrivateFormat(x3->GetStorageFormat()) || IsPrivateFormat(y->GetStorageFormat())),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Private format is not supported, only ND is supported: x1=%d, x2=%d, x3=%d, y=%d.",
            static_cast<int>(x1->GetStorageFormat()), static_cast<int>(x2->GetStorageFormat()),
            static_cast<int>(x3->GetStorageFormat()), static_cast<int>(y->GetStorageFormat())),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, const aclTensor* y)
{
    // 1. 检查参数是否为空指针。
    CHECK_RET(CheckNotNull(x1, x2, x3, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查 dtype（在支持集内 + x1/x2/x3/y 一致）。
    CHECK_RET(CheckDtypeValid(x1, x2, x3, y), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查 shape（x1/x2/y 一致、x3 单元素、最大维度限制）。
    CHECK_RET(CheckShape(x1, x2, x3, y), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查 format（仅 ND，不支持私有格式）。
    CHECK_RET(CheckFormat(x1, x2, x3, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 *
 * 标准流程：
 * 1. CREATE_EXECUTOR()  - 创建执行器
 * 2. CheckParams()      - 参数检查
 * 3. 空 Tensor 快速返回
 * 4. Contiguous()       - 非连续 Tensor 转连续（保证 Kernel 输入连续）
 * 5. l0op::FusedMulAddN - 调用 L0 算子（InferShape + 调度到已注册的 FusedMulAddN aicore）
 * 6. ViewCopy()         - 输出非连续处理（支持用户 y tensor 任意 stride）
 * 7. GetWorkspaceSize() - 获取 workspace 大小
 */
aclnnStatus aclnnFusedMulAddNGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedMulAddN, DFX_IN(x1, x2, x3), DFX_OUT(y));

    // 固定写法，创建 OpExecutor。
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查。
    auto ret = CheckParams(x1, x2, x3, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空 Tensor 快速返回（x1/x2/y 同 shape，任一为空即整体为空；x3 单元素恒非空）。
    if (x1->IsEmpty() || x2->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的 tensor。
    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto x3Contiguous = l0op::Contiguous(x3, uniqueExecutor.get());
    CHECK_RET(x3Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用 L0 算子：y = x1 * x3[0] + x2。
    const aclTensor* opResult = l0op::FusedMulAddN(x1Contiguous, x2Contiguous, x3Contiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出 y 上，y 可能是非连续的 tensor。
    auto viewCopyResult = l0op::ViewCopy(opResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的 workspace 大小。
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 把 uniqueExecutor 持有的 executor 转移给 executor
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 */
aclnnStatus aclnnFusedMulAddN(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedMulAddN);
    // 固定写法，调用框架能力，完成计算。
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
