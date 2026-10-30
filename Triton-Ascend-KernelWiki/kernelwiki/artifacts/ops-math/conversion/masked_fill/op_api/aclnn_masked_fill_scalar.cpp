/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_masked_fill_scalar.h"
#include "masked_fill.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "conversion/squeeze/op_host/op_api/squeeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* MaskedFill.Scalar 算子的完整计算流程如下:
 *       selfRef                     mask                       value
 *         |                           |                          |
 *         |                           |                          |
 * Contiguous(workspace_0)    Contiguous(workspace_2)             |
 *           \                         |                          |
 *     Unsqueeze(workspace_1)     Cast(workspace_3)         Cast(workspace_4)
 *               \                     |                         /
 *                 \                   |                       /
 *                              MaskedFill(workspace_5)
 *                                     |
 *                            Squeeze(workspace_6)
 *                                     |
 *                                  ViewCopy
 *                                     |
 *                                   result
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* selfRef, const aclTensor* mask, const aclScalar* value)
{
    OP_CHECK_NULL(selfRef, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(value, return false);
    return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor* selfRef, const aclTensor* mask)
{
    // 如果soc是1980芯片，则不支持DT_BF16，需要校验拦截
    if (!CheckSocVersionIsSupportBf16() && (selfRef->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Input dtype of aclnnInplaceMaskedFillScalar is not support bfloat16 in current socversion.");
        return false;
    }

    // 检查self的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, DTYPE_SUPPORT_LIST, return false);

    // 检查mask的数据类型是否为bool
    OP_CHECK_DTYPE_NOT_MATCH(mask, op::DataType::DT_BOOL, return false);

    return true;
}

static bool CheckShape(const aclTensor* selfRef, const aclTensor* mask)
{
    const int MAX_DIM = 8;
    OP_CHECK_MAX_DIM(selfRef, MAX_DIM, return false);
    OP_CHECK_MAX_DIM(mask, MAX_DIM, return false);

    Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(selfRef, mask, broadcastShape, return false);

    if (broadcastShape != selfRef->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(selfRef->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* selfRef, const aclTensor* mask, const aclScalar* value)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(selfRef, mask, value), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(selfRef, mask), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查双输入是否能broadcast
    CHECK_RET(CheckShape(selfRef, mask), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceMaskedFillScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* mask, const aclScalar* value, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceMaskedFillScalar, DFX_IN(selfRef, mask, value), DFX_OUT(selfRef));
    // 固定写法，参数检查
    auto ret = CheckParams(selfRef, mask, value);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (selfRef->IsEmpty() || mask->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入mask转换成连续的tensor
    auto maskContiguous = l0op::Contiguous(mask, uniqueExecutor.get());
    CHECK_RET(maskContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入mask的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto maskCasted = l0op::Cast(maskContiguous, DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(maskCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* valueTensor = uniqueExecutor->ConvertToTensor(value, selfRef->GetDataType());

    // 固定写法，将计算结果转换成输出out的数据类型
    auto maskfillOut = l0op::MaskedFill(selfContiguous, maskCasted, valueTensor, uniqueExecutor.get());
    CHECK_RET(maskfillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(maskfillOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceMaskedFillScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceMaskedFillScalar);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
