/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_frac.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "sub.h"
#include "math/zero_op/op_api/zero_op.h"
#include "math/trunc/op_api/trunc.h"
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,   DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT8,  DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,   DataType::DT_UINT8, DataType::DT_BF16};

// 根据API定义，列出需要调用Trunc算子的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_NEED_TRUNC_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> ASCEND910B_NEED_TRUNC_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

// 检查输入和输出是否是空指针
static inline bool CheckNotNull(const aclTensor* input, const aclTensor* out)
{
    // 检查输入是否是空指针
    OP_CHECK_NULL(input, return false);

    // 检查输入是否是空指针
    OP_CHECK_NULL(out, return false);

    return true;
}

// 获取芯片类型，判断芯片是否为Ascend910B
static bool CheckIsAscend910BSocVersion()
{
    return (
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        IsRegBase());
}

static bool CheckDtypeValid(const aclTensor* input, const aclTensor* out)
{
    const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST =
        CheckIsAscend910BSocVersion() ? ASCEND910B_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_SUPPORT_LIST;

    // 检查输入和输出的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(out, input->GetDataType(), return false);

    // 检查输入和输出的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(input, CURRENT_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckShape(const aclTensor* input, const aclTensor* out)
{
    // input和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(input, out, return false);

    // input的维度必须小于等于8
    OP_CHECK_MAX_DIM(input, MAX_SUPPORT_DIMS_NUMS, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入和输出的数据类型是否满足约束，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(input, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入和输出的shape是否满足约束
    CHECK_RET(CheckShape(input, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecFracGetWorkspaceSize(
    const aclTensor* input, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(input, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input->IsEmpty()) {
        // 根据实际情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const std::initializer_list<op::DataType> CURRENT_NEED_TRUNC_SUPPORT_LIST =
        CheckIsAscend910BSocVersion() ? ASCEND910B_NEED_TRUNC_SUPPORT_LIST : ASCEND910_NEED_TRUNC_SUPPORT_LIST;
    const aclTensor* fracOutRet = nullptr;
    if (CheckType(input->GetDataType(), CURRENT_NEED_TRUNC_SUPPORT_LIST)) {
        auto inputOtherInteger = l0op::Trunc(inputContiguous, uniqueExecutor.get());
        CHECK_RET(inputOtherInteger != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 执行L0 Sub算子
        fracOutRet = l0op::Sub(inputContiguous, inputOtherInteger, uniqueExecutor.get());
    } else {
        // 执行L0 ZerosLike算子
        fracOutRet = l0op::ZerosLike(inputContiguous, uniqueExecutor.get());
    }
    CHECK_RET(fracOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上
    auto viewCopyResult = l0op::ViewCopy(fracOutRet, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把uniqueExecutor持有executor转移个executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFracGetWorkspaceSize(
    const aclTensor* input, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnFrac, DFX_IN(input), DFX_OUT(out));
    return ExecFracGetWorkspaceSize(input, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceFracGetWorkspaceSize(aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceFrac, DFX_IN(inputRef), DFX_OUT(inputRef));
    return ExecFracGetWorkspaceSize(inputRef, inputRef, workspaceSize, executor);
}

aclnnStatus aclnnFrac(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnFrac);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceFrac(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceFrac);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
