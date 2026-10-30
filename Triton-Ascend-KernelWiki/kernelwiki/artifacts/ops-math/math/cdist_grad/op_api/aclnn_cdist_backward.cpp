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
 * \file aclnn_cdist_backward.cpp
 * \brief
 */
#include "aclnn_cdist_backward.h"
#include "cdist_grad.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_SUPPORT_DIM = 7;
static const int64_t MIN_SUPPORT_DIM = 2;
static const int64_t NUMBER_TWO = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckNotNull(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, aclTensor* out)
{
    OP_CHECK_NULL(grad, return false);
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(cdist, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(grad, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(cdist, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

    if (grad->GetDataType() != x1->GetDataType() || grad->GetDataType() != x2->GetDataType() ||
        grad->GetDataType() != cdist->GetDataType() || grad->GetDataType() != out->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "All tensors are expected to share the same datatype.");
        return false;
    }

    return true;
}

static bool CheckDims(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, aclTensor* out)
{
    OP_CHECK_MAX_DIM(grad, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MIN_DIM(grad, MIN_SUPPORT_DIM, return false);
    auto gradDimNum = grad->GetViewShape().GetDimNum();
    if ((gradDimNum != x1->GetViewShape().GetDimNum()) || (gradDimNum != x2->GetViewShape().GetDimNum()) ||
        (gradDimNum != cdist->GetViewShape().GetDimNum()) || (gradDimNum != out->GetViewShape().GetDimNum())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "All tensors are expected to share the same dim num.");
        return false;
    }
    return true;
}

static bool getBroadcastShape(const aclTensor* x1, const aclTensor* x2, op::Shape& broadcastShape)
{
    auto x1Shape = x1->GetViewShape();
    auto x2Shape = x2->GetViewShape();
    auto lastDim = x1Shape.GetDimNum() - 1;
    x1Shape.AppendDim(x1Shape[lastDim]);
    x1Shape.SetDim(lastDim, 1);
    if (lastDim >= x2Shape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimensions of input x1 and x2 must be equal.");
        return false;
    }
    x2Shape.AppendDim(x2Shape[lastDim]);
    x2Shape.SetDim(lastDim, x2Shape[lastDim - 1]);
    x2Shape.SetDim(lastDim - 1, 1);

    if (!BroadcastInferShape(x1Shape, x2Shape, broadcastShape)) {
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, aclTensor* out)
{
    size_t dimNum = grad->GetViewShape().GetDimNum();
    auto cdistShape = cdist->GetViewShape();

    op::Shape broadcastShape;
    if (!getBroadcastShape(x1, x2, broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(x2->GetViewShape()).GetString(),
            op::ToString(x1->GetViewShape()).GetString());
        return false;
    }

    for (size_t i = 0; i < dimNum; i++) {
        if (cdistShape[i] != broadcastShape[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cdist[%lu] : %ld should be equal to %ld .", i, cdistShape[i], broadcastShape[i]);
            return false;
        }
    }
    OP_CHECK_SHAPE_NOT_EQUAL(cdist, grad, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(out, x1, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, float p, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(grad, x1, x2, cdist, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(grad, x1, x2, cdist, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的dim 是否满足
    CHECK_RET(CheckDims(grad, x1, x2, cdist, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入形状是否满足
    CHECK_RET(CheckShape(grad, x1, x2, cdist, out), ACLNN_ERR_PARAM_INVALID);

    if (p != -1.0f && p < 0.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnCdistBackward only supports non-negative p values.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCdistBackwardGetWorkspaceSize(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, float p,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnCdistBackward, DFX_IN(grad, x1, x2, cdist, p), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(grad, x1, x2, cdist, p, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (grad->IsEmpty() || x1->IsEmpty() || x2->IsEmpty() || cdist->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    size_t dimNum = grad->GetViewShape().GetDimNum();
    // 将输转换成连续的tensor
    auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    auto cdistContiguous = l0op::Contiguous(cdist, uniqueExecutor.get());
    CHECK_RET(
        gradContiguous != nullptr && x1Contiguous != nullptr && x2Contiguous != nullptr &&
            cdistContiguous != nullptr,
        ACLNN_ERR_INNER_NULLPTR);
    
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    bool needCast = x1 -> GetDataType() == op::DataType::DT_BF16 && (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93);

    if (needCast) {
        gradContiguous = l0op::Cast(gradContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        x1Contiguous = l0op::Cast(x1Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        x2Contiguous = l0op::Cast(x2Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        cdistContiguous = l0op::Cast(cdistContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(
            gradContiguous != nullptr && x1Contiguous != nullptr && x2Contiguous != nullptr &&
                cdistContiguous != nullptr,
            ACLNN_ERR_INNER_NULLPTR);
    }

    auto gradUnsqueezeNd = l0op::UnsqueezeNd(gradContiguous, dimNum, uniqueExecutor.get());
    auto x1UnsqueezeNd = l0op::UnsqueezeNd(x1Contiguous, dimNum - 1, uniqueExecutor.get());
    auto x2UnsqueezeNd = l0op::UnsqueezeNd(x2Contiguous, dimNum - NUMBER_TWO, uniqueExecutor.get());
    auto cdistUnsqueezeNd = l0op::UnsqueezeNd(cdistContiguous, dimNum, uniqueExecutor.get());

    op::Shape broadcastShape;
    BroadcastInferShape(x1UnsqueezeNd->GetViewShape(), x2UnsqueezeNd->GetViewShape(), broadcastShape);
    FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = ToShapeVector(broadcastShape);
    auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
    CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradBroadcast = l0op::BroadcastTo(gradUnsqueezeNd, broadcastShapeArray, uniqueExecutor.get());
    auto x1Broadcast = l0op::BroadcastTo(x1UnsqueezeNd, broadcastShapeArray, uniqueExecutor.get());
    auto x2Broadcast = l0op::BroadcastTo(x2UnsqueezeNd, broadcastShapeArray, uniqueExecutor.get());
    auto cdistBroadcast = l0op::BroadcastTo(cdistUnsqueezeNd, broadcastShapeArray, uniqueExecutor.get());
    auto result =
        l0op::CdistGrad(gradBroadcast, x1Broadcast, x2Broadcast, cdistBroadcast, p, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (needCast) {
        result = l0op::Cast(result, op::DataType::DT_BF16, uniqueExecutor.get());
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyOutputResult = l0op::ViewCopy(result, out, uniqueExecutor.get());

    CHECK_RET(viewCopyOutputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCdistBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCdistBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif