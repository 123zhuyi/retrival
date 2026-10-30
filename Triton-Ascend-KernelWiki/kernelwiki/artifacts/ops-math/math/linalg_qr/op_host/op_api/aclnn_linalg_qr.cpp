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
 * \file aclnn_linalg_qr.cpp
 * \brief
 */

#include "aclnn_linalg_qr.h"
#include "../../../q_r/op_host/op_api/qr.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const int64_t REDUCE_REDUCED = 0;
static const int64_t REDUCE_COMPLETE = 1;
static const int64_t REDUCE_R = 2;

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *Q, const aclTensor *R)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(Q, return false);
    OP_CHECK_NULL(R, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *Q, const aclTensor *R)
{
    // 检查self的数据类型是否在ScatterAdd算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    // 检查Q的数据类型是否在ScatterAdd算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(Q, DTYPE_SUPPORT_LIST, return false);
    // 检查R的数据类型是否在ScatterAdd算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(R, DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckShape(const aclTensor *self, int64_t mode, const aclTensor *Q, const aclTensor *R)
{
    // 检查 self 维度是否大于等于2， 且小于等于8
    op::Shape selfShape = self->GetViewShape();
    auto dimNum = selfShape.GetDimNum();
    // QR分解的输入至少为2维
    OP_CHECK_MIN_DIM(self, 2, return false);
    OP_CHECK_MAX_DIM(self, 8, return false);
    // m为倒数第2维度，n为倒数第1维度
    auto m = selfShape.GetDim(dimNum - 2);
    auto n = selfShape.GetDim(dimNum - 1);
    auto k = std::min(m, n);
    auto emptyShape = op::Shape{0};
    auto qShape = selfShape;
    auto rShape = selfShape;
    // 当mode为reduced时 Q shape (*, m, k), R shape (*, k, n)
    // 当mode为complete时，Q shape (*, m, m), R shpae (*, m, n)
    // 当mode为r时，Q shape (0), R shape (*, k, n)
    if (mode == REDUCE_R) {
        // 修改q的最后1维
        qShape = emptyShape;
        // 修改r的倒数第2维
        rShape.SetDim(dimNum - 2, k);
    } else if (mode == REDUCE_COMPLETE) {
        qShape.SetDim(dimNum - 1, m);
    } else {
        // 修改q的最后1维
        qShape.SetDim(dimNum - 1, k);
        // 修改r的倒数第2维
        rShape.SetDim(dimNum - 2, k);
    }
    OP_CHECK(Q->GetViewShape() == qShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "The output Q should have the following shape %s, but got %s instead.",
                     op::ToString(qShape).GetString(), op::ToString(Q->GetViewShape()).GetString()),
             return false);
    OP_CHECK(R->GetViewShape() == rShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "The output R should have the following shape %s, but got %s instead.",
                     op::ToString(rShape).GetString(), op::ToString(R->GetViewShape()).GetString()),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t mode, const aclTensor *Q, const aclTensor *R)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, Q, R), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, Q, R), ACLNN_ERR_PARAM_INVALID);

    // 3. ND 算子不检查格式
    // 4. 检查self和Q,R的shape是否符合约束
    CHECK_RET(CheckShape(self, mode, Q, R), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor *self, int64_t mode, aclTensor *Q, aclTensor *R,
                                          uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnLinalgQr, DFX_IN(self, mode), DFX_OUT(Q, R));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, mode, Q, R);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Qr算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    // 当仅最后一个维度为 0，且some为false时，此时Q不为空，后两维度tensor为 eye
    if (self->IsEmpty() && Q->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (selfContiguous->GetDataType() == op::DataType::DT_FLOAT16) {
        // AICPU does not support with fp16 input
        selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 调用QR算子Kernel
    auto some = (mode == REDUCE_COMPLETE) ? false : true;
    auto outArray = l0op::Qr(selfContiguous, some, uniqueExecutor.get());

    // 获取对应的Q，R
    auto resultQ = std::get<0>(outArray);
    auto resultR = std::get<1>(outArray);
    CHECK_RET(resultQ != nullptr && resultR != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOutQ = l0op::Cast(resultQ, Q->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOutQ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOutR = l0op::Cast(resultR, R->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOutR != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出q, r上，可能是非连续的tensor
    if (mode != REDUCE_R) {
        auto viewCopyResultQ = l0op::ViewCopy(castOutQ, Q, uniqueExecutor.get());
        CHECK_RET(viewCopyResultQ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto viewCopyResultR = l0op::ViewCopy(castOutR, R, uniqueExecutor.get());
    CHECK_RET(viewCopyResultR != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinalgQr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnLinalgQr);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
