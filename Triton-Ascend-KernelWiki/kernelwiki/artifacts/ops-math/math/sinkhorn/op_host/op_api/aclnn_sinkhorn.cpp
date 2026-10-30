/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "aclnn_sinkhorn.h"
#include "sinkhorn.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t IO_DIM_LEN = 2;
constexpr int32_t COST_COL_DIM = 1;
constexpr int32_t MAX_COST_COL = 4096;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST_NOT_SUPPORT_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST_SUPPORT_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8, op::DataType::DT_BOOL};

inline static bool CheckNotNull(const aclTensor* cost, const aclTensor* p)
{
    OP_CHECK_NULL(cost, return false);
    OP_CHECK_NULL(p, return false);
    return true;
}

static const std::initializer_list<op::DataType> CheckSocVersionIsSupportBf16(void)
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return SELF_DTYPE_SUPPORT_LIST_SUPPORT_BF16;
    }
    return SELF_DTYPE_SUPPORT_LIST_NOT_SUPPORT_BF16;
}

static bool CheckDtypeValid(const aclTensor* cost, const aclTensor* p)
{
    auto SELF_DTYPE_SUPPORT_LIST = CheckSocVersionIsSupportBf16();
    // 检查cost的数据类型是否在Sinkhorn算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(cost, SELF_DTYPE_SUPPORT_LIST, return false);
    // 检查p的数据类型是否在Sinkhorn算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(p, SELF_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckShape(const aclTensor* cost, const aclTensor* p)
{
    OP_CHECK_WRONG_DIMENSION(cost, IO_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(cost, p, return false);

    int32_t col = cost->GetViewShape().GetDim(COST_COL_DIM);
    if (col > MAX_COST_COL) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "the column number of cost is %d, it cannot be larger than %d.", col,
            MAX_COST_COL);
        return false;
    }

    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* cost, const aclTensor* p)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(cost, p), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(cost, p), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入形状是否满足
    CHECK_RET(CheckShape(cost, p), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSinkhornGetWorkspaceSize(
    const aclTensor* cost, const aclScalar* tol, aclTensor* p, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSinkhorn, DFX_IN(cost, tol), DFX_OUT(p));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(cost, p);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (cost->IsEmpty() || p->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // cost如果非连续，需要转连续
    auto costContiguous = l0op::Contiguous(cost, uniqueExecutor.get());
    CHECK_RET(costContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (costContiguous->GetDataType() == DataType::DT_FLOAT) {
        // 调用Sinkhorn算子
        l0op::Sinkhorn(costContiguous, tol, p, uniqueExecutor.get());
    } else {
        auto costCast = l0op::Cast(costContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(costCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto pCast = (uniqueExecutor.get())->AllocTensor(cost->GetViewShape(), DataType::DT_FLOAT);
        l0op::Sinkhorn(costCast, tol, pCast, uniqueExecutor.get());
        const aclTensor* pOut = l0op::Cast(pCast, p->GetDataType(), uniqueExecutor.get());
        auto viewCopyResult = l0op::ViewCopy(pOut, p, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSinkhorn(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSinkhorn);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
