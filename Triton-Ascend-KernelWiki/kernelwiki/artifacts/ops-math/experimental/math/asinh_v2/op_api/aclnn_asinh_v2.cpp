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
 * @file aclnn_asinh_v2.cpp
 * @brief ACLNN L2 API 实现 - AsinhV2 算子
 *
 * ACLNN 接口采用两段式设计：
 * 1. aclnnAsinhV2GetWorkspaceSize - 计算 workspace 大小，创建执行器
 * 2. aclnnAsinhV2 - 执行计算
 *
 * 文件组织：
 * - aclnn_asinh_v2.h/cpp  -> L2 API（本文件）：参数检查、Contiguous 处理
 * - asinh_v2.h/cpp        -> L0 API（底层实现）：形状推导、Kernel 调度
 *
 * 支持 dtype：
 * - float16 / float32：直接路径，Kernel 侧直接计算
 * - int8/int16/int32/int64/uint8/bool/double（Cast 路径）：op_api 层 Cast 到 float32 后走 float32 Kernel 路径
 */

#include "aclnn_asinh_v2.h"
#include "asinh_v2.h"
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

// 支持的输入 dtype（L2层）
// float16 / float32：直接路径；int8/int16/int32/int64/uint8/bool/double：Cast 路径（→ float32）
static const std::initializer_list<op::DataType> SUPPORTED_INPUT_DTYPE_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_INT8, DataType::DT_INT16, DataType::DT_INT32,
    DataType::DT_INT64,   DataType::DT_UINT8, DataType::DT_BOOL, DataType::DT_DOUBLE};

// 需要 Cast 到 float32 的 dtype（所有非float16/float32 类型）
static bool NeedsCastToFloat32(DataType dtype)
{
    return dtype == DataType::DT_INT8 || dtype == DataType::DT_INT16 || dtype == DataType::DT_INT32 ||
           dtype == DataType::DT_INT64 || dtype == DataType::DT_UINT8 || dtype == DataType::DT_BOOL ||
           dtype == DataType::DT_DOUBLE;
}

static bool IsDtypeSupported(DataType dtype) { return CheckType(dtype, SUPPORTED_INPUT_DTYPE_LIST); }

// 根据输入 dtype 推导期望的输出 dtype
// float16 -> float16; float32 -> float32; int8/int32/bool/double -> float32
static DataType GetExpectedOutputDtype(DataType inputDtype)
{
    if (inputDtype == DataType::DT_FLOAT16) {
        return DataType::DT_FLOAT16;
    }
    return DataType::DT_FLOAT;
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    if (!IsDtypeSupported(x->GetDataType())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "AsinhV2: input dtype %d not supported. "
            "Supported: FLOAT16, FLOAT, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE.",
            static_cast<int>(x->GetDataType()));
        return false;
    }
    // 输出 dtype 校验：与 GetExpectedOutputDtype 一致
    DataType expectedOutDtype = GetExpectedOutputDtype(x->GetDataType());
    OP_CHECK_DTYPE_NOT_MATCH(out, expectedOutDtype, return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* out)
{
    if (IsPrivateFormat(x->GetStorageFormat()) || IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AsinhV2: private format not supported: x=%d, out=%d",
            static_cast<int>(x->GetStorageFormat()), static_cast<int>(out->GetStorageFormat()));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* out)
{
    if (!CheckNotNull(x, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(x, out)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(x, out)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(x, out)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 *
 * 标准流程：
 * 1. CREATE_EXECUTOR()   - 创建执行器
 * 2. CheckParams()       - 参数检查
 * 3. 空 Tensor 快速返回  - IsEmpty 检查
 * 4. Contiguous()        - 非连续 Tensor 转换
 * 5. Cast（若需要）      - int8/bool → float32 转换
 * 6. l0op::AsinhV2() - 调用 L0 算子
 * 7. ViewCopy()          - 输出非连续处理
 * 8. GetWorkspaceSize()  - 获取 workspace 大小
 */
extern "C" aclnnStatus aclnnAsinhV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAsinhV2, DFX_IN(x), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空 Tensor 快速返回
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Cast 路径：int8/bool → float32
    const aclTensor* xForKernel = xContiguous;
    if (NeedsCastToFloat32(x->GetDataType())) {
        xForKernel = l0op::Cast(xContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(xForKernel != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* opResult = l0op::AsinhV2(xForKernel, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 */
extern "C" aclnnStatus aclnnAsinhV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAsinhV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
