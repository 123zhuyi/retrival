/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_replication_pad1d.h"
#include "padv3.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "conversion/squeeze/op_host/op_api/squeeze.h"
#include "op_replication_pad.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckShape(const aclTensor* self, const aclIntArray* padding, const aclTensor* out)
{
    auto selfDimnum = self->GetViewShape().GetDimNum();
    // self只支持2维和3维
    OP_CHECK_MIN_DIM(self, 2, return false);
    OP_CHECK_MAX_DIM(self, 3, return false);

    // self, out维度需要一致
    OP_CHECK(selfDimnum == out->GetViewShape().GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self, out dim should be same."), return false);

    // padding长度为2
    OP_CHECK(padding->Size() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 2, but got %lu.", padding->Size()),
             return false);

    op::Shape expectShape;
    expectShape.SetDimNum(selfDimnum);
    size_t paddingDim = 1;
    if (selfDimnum > paddingDim) {
        size_t dimToCompare = selfDimnum - paddingDim;
        for (size_t i = 0; i < dimToCompare; i++) {
            expectShape.SetDim(i, self->GetViewShape().GetDim(i));
        }
    }
    expectShape.SetDim(selfDimnum - 1, self->GetViewShape().GetDim(selfDimnum - 1) + (*padding)[0] + (*padding)[1]);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* padding, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, padding, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, padding, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

inline static aclnnStatus InputPreprocess(const aclTensor*& self, int64_t dimCp, aclOpExecutor* executor)
{
    // 如果非连续，需要转连续
    self = l0op::Contiguous(self, executor);
    CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 2 is dim num
    if (dimCp == 2) {
        // 0 is index
        const int64_t appendDim[] = {0};
        // 1 is the dim num to be unsqueezed
        aclIntArray* dimArray = executor->AllocIntArray(appendDim, 1);
        self = l0op::UnsqueezeNd(self, dimArray, executor);
        CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReplicationPad1dGetWorkspaceSize(const aclTensor* self, const aclIntArray* padding, aclTensor* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnReplicationPad1d, DFX_IN(self, padding), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, padding, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0UL;
        // 2 is dim num
        if (self->GetViewShape().GetDimNum() == 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected 2D or 3D tensor with possibly 0 batch size and other non-zero dimentions for input.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 3 is dim num
        if (self->GetViewShape().GetDimNum() == 3) {
            // 1, 2 are indexes
            if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "Expected 2D or 3D tensor with possibly 0 batch size and other non-zero dimentions for input.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用l0算子进行计算
    auto dim = self->GetViewShape().GetDimNum();
    auto dimCp = dim;
    ret = InputPreprocess(self, dimCp, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    dim = self->GetViewShape().GetDimNum();
    auto paddingsTensor = GetPaddingTensor(dim, padding, uniqueExecutor.get());
    const aclTensor* pad1dResult = nullptr;
    if (IsRegBase()) {
        pad1dResult = l0op::PadV3(self, paddingsTensor, nullptr, REPLICATION_MODE, true, uniqueExecutor.get());
    } else {
        aclScalar* constantValueScalar = (uniqueExecutor.get())->AllocScalar(0);
        CHECK_RET(constantValueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto constantValueTensor = (uniqueExecutor.get())->ConvertToTensor(constantValueScalar, self->GetDataType());
        pad1dResult = l0op::PadV3(self, paddingsTensor, constantValueTensor, REPLICATION_MODE, true,
                                  uniqueExecutor.get());
    }
    CHECK_RET(pad1dResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 2 is dim num
    if (dimCp == 2) {
        const int64_t appendDim[] = {0};
        aclIntArray* dimArray = (uniqueExecutor.get())->AllocIntArray(appendDim, 1);
        pad1dResult = l0op::SqueezeNd(pad1dResult, dimArray, uniqueExecutor.get());
        CHECK_RET(pad1dResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(pad1dResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReplicationPad1d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReplicationPad1d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
