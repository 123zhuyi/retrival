/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_atanh.h"
#include "atanh.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST_ATANH = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_DOUBLE, DataType::DT_INT8,
    DataType::DT_INT16,     DataType::DT_INT32,      DataType::DT_INT64,  DataType::DT_BOOL,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST_ATANH = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_DOUBLE, DataType::DT_INT8,
    DataType::DT_INT16,     DataType::DT_INT32,      DataType::DT_INT64,  DataType::DT_BOOL,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_UINT8,  DataType::DT_BF16};

static const std::initializer_list<DataType> OUTPUT_DTYPE_SUPPORT_LIST_ATANH = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_DOUBLE,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_BF16};

// 根据API定义，列出需要将输入Tensor转换为FLOAT类型的所有dtype
static const std::initializer_list<DataType> NEED_CAST_DTYPE_LIST_ATANH = {DataType::DT_INT8,  DataType::DT_INT16,
                                                                           DataType::DT_INT32, DataType::DT_INT64,
                                                                           DataType::DT_BOOL,  DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910_DTYPE_SELFREF_LIST_ATANH = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

static bool CheckDtypeValid(const aclTensor* input, const aclTensor* out)
{
    // 检查输入的数据类型是否在算子的支持列表内
    auto supportList =
        GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST_ATANH, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST_ATANH);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, supportList, return false);

    // 检查输出的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUTPUT_DTYPE_SUPPORT_LIST_ATANH, return false);

    return true;
}

static bool CheckInplaceDtypeValid(aclTensor* selfRef)
{
    auto inplaceSupportList =
        GetDtypeSupportListV2(OUTPUT_DTYPE_SUPPORT_LIST_ATANH, ASCEND910_DTYPE_SELFREF_LIST_ATANH);
    // 检查selfRef的数据类型是否在inplace atanh算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static aclnnStatus CheckParamsAtanh(const aclTensor* input, const aclTensor* out)
{
    // 检查输入和输出的数据类型是否满足约束，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(input, out), ACLNN_ERR_PARAM_INVALID);
    // 检查输入和输出的shape是否满足约束
    CHECK_RET(CheckSameShape1In1Out(input, out), ACLNN_ERR_PARAM_INVALID);

    if (input->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGW("Only support ND format for atanh/inplaceAtanh operator.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParamsAtanh(aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace atanh算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecAtanhGetWorkspaceSize(
    const aclTensor* input, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_NOT_NULL(input, out);
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParamsAtanh(input, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用cast算子将不支持的类型转换为FLOAT类型
    auto castDtype = inputContiguous->GetDataType();
    if (CheckType(castDtype, NEED_CAST_DTYPE_LIST_ATANH)) {
        castDtype = DataType::DT_FLOAT;
    }

    auto inputCast = l0op::Cast(inputContiguous, castDtype, uniqueExecutor.get());
    CHECK_RET(inputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行L0算子
    auto atanhOutRet = l0op::Atanh(inputCast, uniqueExecutor.get());
    CHECK_RET(atanhOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(atanhOutRet, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAtanhGetWorkspaceSize(
    const aclTensor* input, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAtanh, DFX_IN(input), DFX_OUT(out));
    return ExecAtanhGetWorkspaceSize(input, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceAtanhGetWorkspaceSize(aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceAtanh, DFX_IN(inputRef), DFX_OUT(inputRef));
    auto ret = CheckInplaceParamsAtanh(inputRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecAtanhGetWorkspaceSize(inputRef, inputRef, workspaceSize, executor);
}

aclnnStatus aclnnAtanh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnAtanh);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAtanh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceAtanh);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
