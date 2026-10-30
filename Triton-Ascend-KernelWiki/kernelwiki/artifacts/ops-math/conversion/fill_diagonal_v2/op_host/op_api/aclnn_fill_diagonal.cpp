/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_fill_diagonal.cpp
 * \brief
 */
#include "aclnn_fill_diagonal.h"
#include "fill_diagonal.h"
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
#include "opdev/make_op_executor.h"
#include "op_api/op_api_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MIN_DIM_LEN = 2;
constexpr size_t BN_MIN_SUPPORT_DIMS_NUMS_FOR_FILL_DIAGONAL = 2;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,    op::DataType::DT_INT16, op::DataType::DT_INT32,
    op::DataType::DT_INT64,   op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_INT16,   op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,    op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* selfRef, const aclScalar* fillValue)
{
    OP_CHECK_NULL(selfRef, return false);
    OP_CHECK_NULL(fillValue, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* selfRef)
{
    // 根据芯片类型获取数据类型支持列表
    bool isAscend910BSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E);
    const std::initializer_list<op::DataType> dtypeSupportList =
        isAscend910BSocVersion ? ASCEND910B_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_SUPPORT_LIST;

    // 检查selfRef的数据类型是否在FillDiagonal算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, dtypeSupportList, return false);
    return true;
}

static bool IsSameDimLength(const aclTensor* selfRef)
{
    size_t dimSize = selfRef->GetViewShape().GetDimNum();
    auto dimLength = selfRef->GetViewShape().GetDim(0);
    for (size_t i = 1; i < dimSize; i++) {
        if (dimLength != selfRef->GetViewShape().GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* selfRef)
{
    OP_CHECK_MAX_DIM(selfRef, MAX_SUPPORT_DIMS_NUMS, return false);

    // 检查selfRef维度是否大于等于2
    OP_CHECK_MIN_DIM(selfRef, BN_MIN_SUPPORT_DIMS_NUMS_FOR_FILL_DIAGONAL, return false);

    // 检查selfRef维度是否大于2且各维度长度相等
    if (selfRef->GetViewShape().GetDimNum() > MIN_DIM_LEN && !IsSameDimLength(selfRef)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dim of selfRef[%zu] greater than %zu, all dimensions of input must be of equal length.",
            selfRef->GetViewShape().GetDimNum(), MIN_DIM_LEN);
        return false;
    }
    return true;
}

static bool CheckCanCastType(const aclScalar* fillValue)
{
    // 检查fillValue是否能转换为FLOAT
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(fillValue->GetDataType(), op::DataType::DT_FLOAT, return false);
    return true;
}

static bool CheckNotOverflow(const aclTensor* selfRef, const aclScalar* fillValue)
{
    auto dataType = selfRef->GetDataType();
    int8_t overFlowFlag = 1;
    int8_t floatFlag = 1;
    int8_t intFlag = 2;

    switch (dataType) {
        case op::DataType::DT_FLOAT: {
            overFlowFlag = fillValue->CheckOverflows<float>() ? overFlowFlag << floatFlag : overFlowFlag;
            break;
        }
        case op::DataType::DT_FLOAT16: {
            overFlowFlag = fillValue->CheckOverflows<op::fp16_t>() ? overFlowFlag << floatFlag : overFlowFlag;
            break;
        }
        case op::DataType::DT_DOUBLE: {
            overFlowFlag = fillValue->CheckOverflows<double>() ? overFlowFlag << floatFlag : overFlowFlag;
            break;
        }
        case op::DataType::DT_INT8: {
            overFlowFlag = fillValue->CheckOverflows<int8_t>() ? overFlowFlag << intFlag : overFlowFlag;
            break;
        }
        case op::DataType::DT_INT16: {
            overFlowFlag = fillValue->CheckOverflows<int16_t>() ? overFlowFlag << intFlag : overFlowFlag;
            break;
        }
        case op::DataType::DT_INT32: {
            overFlowFlag = fillValue->CheckOverflows<int32_t>() ? overFlowFlag << intFlag : overFlowFlag;
            break;
        }
        case op::DataType::DT_INT64: {
            overFlowFlag = fillValue->CheckOverflows<int64_t>() ? overFlowFlag << intFlag : overFlowFlag;
            break;
        }
        case op::DataType::DT_UINT8: {
            overFlowFlag = fillValue->CheckOverflows<uint8_t>() ? overFlowFlag << intFlag : overFlowFlag;
            break;
        }
        default: {
            return true;
        }
    }

    if ((overFlowFlag >> floatFlag) == 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "value cannot be converted to type %s without overflow : %lf.",
            op::ToString(dataType).GetString(), fillValue->ToDouble());
    } else if ((overFlowFlag >> intFlag) == 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "value cannot be converted to type %s without overflow : %ld.",
            op::ToString(dataType).GetString(), fillValue->ToInt64());
    }
    return overFlowFlag == 1;
}

static void CheckFormat(const aclTensor* x)
{
    op::Format format = x->GetStorageFormat();
    if (format == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("Format of input gets [%s], this format mat lead to precision failure",
        op::ToString(format).GetString());
    }
}

static aclnnStatus CheckParams(const aclTensor* selfRef, const aclScalar* fillValue)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(selfRef, fillValue), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查是否shape一致
    CHECK_RET(CheckShapeValid(selfRef), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查fillValue是否能转换为FLOAT
    CHECK_RET(CheckCanCastType(fillValue), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查是否溢出
    CHECK_RET(CheckNotOverflow(selfRef, fillValue), ACLNN_ERR_PARAM_INVALID);

    CheckFormat(selfRef);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceFillDiagonalGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* fillValue, bool wrap, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceFillDiagonal, DFX_IN(selfRef, fillValue, wrap), DFX_OUT());

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(selfRef, fillValue);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // FillDiagonal算子的空tensor在kernel中支持
    if (selfRef->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRefContiguous转换成连续的tensor
    auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用FillDiagonal算子kernel
    auto selfRefFilled = l0op::FillDiagonal(selfRefContiguous, fillValue, wrap, uniqueExecutor.get());
    CHECK_RET(selfRefFilled != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(selfRefFilled, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceFillDiagonal(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFillDiagonal);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif