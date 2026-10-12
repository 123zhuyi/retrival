/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_cos.h"
#include "cos.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "common/inc/op_api/op_api_def.h"
#include "common/inc/op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出Ascend910所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_DOUBLE, DataType::DT_INT8,
    DataType::DT_INT16,     DataType::DT_INT32,      DataType::DT_INT64,  DataType::DT_BOOL,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

// 根据API定义，需要列出Ascend910B所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910B_INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_DOUBLE, DataType::DT_INT8,
    DataType::DT_INT16,     DataType::DT_INT32,      DataType::DT_INT64,  DataType::DT_BOOL,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_UINT8,  DataType::DT_BF16};

static const std::initializer_list<DataType> ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_DOUBLE,
    DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_BF16};

// 根据API定义，列出需要将输入Tensor转换为FLOAT类型的所有dtype
static const std::initializer_list<DataType> NEED_CAST_DTYPE_LIST = {DataType::DT_INT8,  DataType::DT_INT16,
                                                                     DataType::DT_INT32, DataType::DT_INT64,
                                                                     DataType::DT_BOOL,  DataType::DT_UINT8};

// 检查输入和输出的数据类型是否在算子的支持列表内
static bool CheckDtypeValid(const aclTensor* input, const aclTensor* out)
{
    // 获取芯片类型，判断芯片是否为Ascend910B
    bool isAscend910BSocVersion =
        (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 ||
         GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510);
    const std::initializer_list<op::DataType> CURRENT_INPUT_DTYPE_SUPPORT_LIST =
        isAscend910BSocVersion ? ASCEND910B_INPUT_DTYPE_SUPPORT_LIST : ASCEND910_INPUT_DTYPE_SUPPORT_LIST;
    const std::initializer_list<op::DataType> CURRENT_OUTPUT_DTYPE_SUPPORT_LIST =
        isAscend910BSocVersion ? ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST : ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST;

    // 检查输入的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(input, CURRENT_INPUT_DTYPE_SUPPORT_LIST, return false);

    // 检查输出的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, CURRENT_OUTPUT_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckInplaceDtypeValid(aclTensor* selfRef)
{
    auto inplaceSupportList =
        GetDtypeSupportListV2(ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST, ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST);
    // 检查selfRef的数据类型是否在inplace cos算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

static aclnnStatus CheckParamsCos(const aclTensor* input, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(input, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(input, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的数据的值是否合理
    CHECK_RET(CheckSameShape1In1Out(input, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParamsCos(aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace cos算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecCosGetWorkspaceSize(
    const aclTensor* input, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParamsCos(input, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* inputCast = nullptr;
    if (CheckType(inputContiguous->GetDataType(), NEED_CAST_DTYPE_LIST)) {
        inputCast = l0op::Cast(inputContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(inputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        inputCast = inputContiguous;
    }
    // 执行L0算子
    auto cosOutRet = l0op::Cos(inputCast, uniqueExecutor.get());
    CHECK_RET(cosOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(cosOutRet, out->GetDataType(), uniqueExecutor.get());
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

aclnnStatus aclnnCosGetWorkspaceSize(
    const aclTensor* input, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCos, DFX_IN(input), DFX_OUT(out));
    return ExecCosGetWorkspaceSize(input, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceCosGetWorkspaceSize(aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceCos, DFX_IN(inputRef), DFX_OUT(inputRef));
    auto out = const_cast<aclTensor*>(inputRef);
    auto ret = CheckInplaceParamsCos(inputRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecCosGetWorkspaceSize(inputRef, out, workspaceSize, executor);
}

aclnnStatus aclnnCos(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnCos);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceCos(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceCos);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
