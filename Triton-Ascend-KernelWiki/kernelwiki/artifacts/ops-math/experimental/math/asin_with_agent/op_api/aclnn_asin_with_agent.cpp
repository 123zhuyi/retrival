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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_asin_with_agent.cpp
 * @brief ACLNN L2 API 实现 - AsinWithAgent 算子
 *
 * 两段式设计：
 *   1. aclnnAsinWithAgentGetWorkspaceSize - 参数检查、Contiguous/ViewCopy 处理、Kernel 调度
 *   2. aclnnAsinWithAgent                 - 执行计算
 *
 * 输出 dtype 规则：
 *   - FLOAT/FLOAT16/DOUBLE -> 与输入相同
 *   - INT8/INT16/INT32/INT64/UINT8/BOOL -> FLOAT32
 *
 * DOUBLE 路径特殊处理（穿刺验证结论：arch32 AICore 禁止 fp64 算术）：
 *   1. 调用 aclCastToFloat32（aclnn 框架提供）将 fp64 输入 tensor 在 Host/NPU 侧转为 fp32
 *   2. 调用 l0op::AsinWithAgent 对 fp32 tensor 执行 Asin
 *   3. 调用 aclCastFromFloat32 将 fp32 结果转回 fp64，写入输出 tensor
 *
 * 迭代二：激活全部 9 种 dtype
 */

#include "aclnn_asin_with_agent.h"
#include "asin_with_agent.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

// 迭代二：激活全部 9 种 dtype
static const std::initializer_list<op::DataType> SUPPORTED_INPUT_DTYPES = {
    DataType::DT_FLOAT,   // TilingKey=0
    DataType::DT_FLOAT16, // TilingKey=1
    DataType::DT_DOUBLE,  // TilingKey=2（Host 端转换）
    DataType::DT_INT8,    // TilingKey=3
    DataType::DT_INT16,   // TilingKey=4
    DataType::DT_INT32,   // TilingKey=5
    DataType::DT_INT64,   // TilingKey=6
    DataType::DT_UINT8,   // TilingKey=7
    DataType::DT_BOOL,    // TilingKey=8
};

// 计算输出 dtype（整数/BOOL 输入 -> FLOAT32 输出）
static op::DataType GetExpectedOutputDtype(op::DataType inputDtype)
{
    switch (inputDtype) {
        case DataType::DT_FLOAT:
            return DataType::DT_FLOAT;
        case DataType::DT_FLOAT16:
            return DataType::DT_FLOAT16;
        case DataType::DT_DOUBLE:
            return DataType::DT_DOUBLE;
        case DataType::DT_INT8:
        case DataType::DT_INT16:
        case DataType::DT_INT32:
        case DataType::DT_INT64:
        case DataType::DT_UINT8:
        case DataType::DT_BOOL:
            return DataType::DT_FLOAT;
        default:
            return DataType::DT_FLOAT;
    }
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y)
{
    if (!CheckType(x->GetDataType(), SUPPORTED_INPUT_DTYPES)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype not supported: dtype=%d.", static_cast<int>(x->GetDataType()));
        return false;
    }
    op::DataType expectedOutDtype = GetExpectedOutputDtype(x->GetDataType());
    if (y->GetDataType() != expectedOutDtype) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Output dtype mismatch: expected=%d, actual=%d",
            static_cast<int>(expectedOutDtype), static_cast<int>(y->GetDataType()));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* y)
{
    auto formatX = x->GetStorageFormat();
    auto formatY = y->GetStorageFormat();
    if (IsPrivateFormat(formatX) || IsPrivateFormat(formatY)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Private format not supported: x=%d, y=%d", static_cast<int>(formatX),
            static_cast<int>(formatY));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(y, ACLNN_MAX_SHAPE_RANK, return false);

    // 校验总元素数不超过 UINT32_MAX，避免 Tiling 侧 int64_t->uint32_t 截断
    int64_t numElem = x->GetViewShape().GetShapeSize();
    if (numElem > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Total element count exceeds UINT32_MAX.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* y)
{
    if (!CheckNotNull(x, y)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(x, y)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(x, y)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(x, y)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 *
 * 标准流程：
 * 1. CREATE_EXECUTOR()  - 创建执行器
 * 2. CheckParams()      - 参数检查
 * 3. 空 Tensor 快速返回
 * 4. Contiguous()       - 非连续 Tensor 转换
 * 5. [DOUBLE 特殊处理]  - Cast fp64->fp32（Host 端，通过 l0op::Cast）
 * 6. l0op::AsinWithAgent() - 调用 L0 算子（DOUBLE 路径接收 fp32）
 * 7. [DOUBLE 特殊处理]  - Cast fp32->fp64（通过 l0op::Cast）
 * 8. ViewCopy()         - 输出非连续处理
 * 9. GetWorkspaceSize() - 获取 workspace 大小
 */
extern "C" aclnnStatus aclnnAsinWithAgentGetWorkspaceSize(
    const aclTensor* x, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAsinWithAgent, DFX_IN(x), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空 tensor 快速返回
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 非连续 tensor 转换
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = nullptr;

    if (x->GetDataType() == DataType::DT_DOUBLE) {
        // DOUBLE 路径：Host 端 fp64->fp32 转换
        // 穿刺验证结论：arch32 AICore 禁止 fp64 算术，必须在 Host 端（通过 NPU Cast 算子）转换
        // 通过 l0op::Cast 将 fp64 tensor 转为 fp32
        auto xFp32 = l0op::Cast(xContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(xFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 对 fp32 tensor 执行 Asin
        const aclTensor* asinFp32Result = l0op::AsinWithAgent(xFp32, uniqueExecutor.get());
        CHECK_RET(asinFp32Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将 fp32 结果 Cast 回 fp64
        opResult = l0op::Cast(asinFp32Result, DataType::DT_DOUBLE, uniqueExecutor.get());
        CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 非 DOUBLE 路径：直接调用 L0 API
        opResult = l0op::AsinWithAgent(xContiguous, uniqueExecutor.get());
        CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 处理输出非连续情况
    auto viewCopyResult = l0op::ViewCopy(opResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 */
extern "C" aclnnStatus aclnnAsinWithAgent(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAsinWithAgent);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
