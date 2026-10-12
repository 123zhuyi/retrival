/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_dropout_v3_grad.cpp
 * \brief
 */

#include "aclnn_dropout_v3_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "dropout_v3_grad.h"
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

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
static const int64_t BIT_NUMBER = 128;
static const int64_t UINT8_BIT_NUMBER = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};

// 对外接口mask仅支持uint8
static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {op::DataType::DT_UINT8};

// 本算子仅在Ascend910_95上提供kernel实现，其余产品形态直接返回错误
static inline bool CheckSocVersion()
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnDropoutV3Grad is not supported in current socversion.");
        return false;
    }
    return true;
}

static inline bool CheckNotNull(const aclTensor* gradY, const aclTensor* mask, const aclTensor* gradX)
{
    OP_CHECK_NULL(gradY, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(gradX, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradY, const aclTensor* mask, const aclTensor* gradX)
{
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(gradY->GetDataType(), gradX->GetDataType(), return false);

    // 检查gradY的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradY, DTYPE_SUPPORT_LIST, return false);

    // 检查mask的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(mask, MASK_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor* gradY, const aclTensor* mask, const aclTensor* gradX)
{
    OP_CHECK_MAX_DIM(gradY, MAX_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradY, gradX, return false);

    // mask元素个数需为align(gradY元素个数,128)/8
    int64_t gradSize = gradY->GetViewShape().GetShapeSize();
    int64_t maskSize = mask->GetViewShape().GetShapeSize();
    int64_t needMaskSize = (gradSize + BIT_NUMBER - 1) / BIT_NUMBER * BIT_NUMBER / UINT8_BIT_NUMBER;
    if (maskSize != needMaskSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of mask has to be %ld, but current is %ld.", needMaskSize, maskSize);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor* gradY, const aclTensor* mask, aclTensor* gradX)
{
    // 1. 检查当前产品形态是否支持
    CHECK_RET(CheckSocVersion(), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradY, mask, gradX), ACLNN_ERR_PARAM_NULLPTR);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradY, mask, gradX), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入输出shape。scale对标torch不做范围校验，任意值直接照乘
    CHECK_RET(CheckShape(gradY, mask, gradX), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDropoutV3GradGetWorkspaceSize(const aclTensor* gradY, const aclTensor* mask, double scale,
                                               aclTensor* gradX, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDropoutV3Grad, DFX_IN(gradY, mask, scale), DFX_OUT(gradX));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradY, mask, gradX);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradY->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor
    auto gradYContiguous = l0op::Contiguous(gradY, uniqueExecutor.get());
    CHECK_RET(gradYContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto maskContiguous = l0op::Contiguous(mask, uniqueExecutor.get());
    CHECK_RET(maskContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // scale为double标量，对内转换成DT_FLOAT的tensor直传kernel
    FVector<float> scaleVector = {static_cast<float>(scale)};
    auto scaleTensor = uniqueExecutor.get()->ConvertToTensor(scaleVector.data(), scaleVector.size(),
                                                             op::DataType::DT_FLOAT);
    CHECK_RET(scaleTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行dropout反向计算
    auto dropOutGradOut = l0op::DropoutV3Grad(gradYContiguous, maskContiguous, scaleTensor, uniqueExecutor.get());
    CHECK_RET(dropOutGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出gradX的数据类型
    auto castOut = l0op::Cast(dropOutGradOut, gradX->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出gradX上，gradX可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, gradX, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDropoutV3Grad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDropoutV3Grad);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
