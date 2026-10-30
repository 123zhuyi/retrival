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
 * \file reduce_std_v2.cpp
 * \brief
 */

#include "reduce_std_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceStdV2);
OP_TYPE_REGISTER(ReduceMean);
OP_TYPE_REGISTER(ReduceStdWithMean);
OP_TYPE_REGISTER(ReduceStdV2Update);

const aclTensor* BroadcastTo(const aclTensor* x, const aclIntArray* shape, aclOpExecutor* executor);

const aclTensor* Expand(const aclTensor* self, const aclIntArray* shape, aclOpExecutor* executor)
{
    return BroadcastTo(self, shape, executor);
}

static int64_t MakeWrapDim(int64_t dim, int64_t rank)
{
    if (rank <= 0) {
        rank = 1;
    }
    return dim < 0 ? dim + rank : dim;
}

static aclTensor* AllocReduceOut(const aclTensor* self, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
    return out;
}

static void SetKeepDimShape(const aclTensor* self, const aclIntArray* dim, aclTensor* out)
{
    op::Shape outShape = self->GetViewShape();
    size_t dimNum = outShape.GetDimNum();
    for (uint64_t i = 0; i < dim->Size(); i++) {
        int64_t dimIndex = static_cast<int64_t>((*dim)[i]);
        int64_t dimNew = dimIndex >= 0 ? dimIndex : dimIndex + static_cast<int64_t>(dimNum);
        outShape.SetDim(dimNew, 1);
    }
    out->SetViewShape(outShape);
}

static bool InferReduceMeanShape(const op::Shape& selfShape, const aclIntArray* dim, bool keepdim, op::Shape& outShape)
{
    constexpr size_t maxMaskLen = 64;
    uint64_t mask[maxMaskLen] = {0};
    if (dim->Size() == 0) {
        for (size_t i = 0; i < maxMaskLen; i++) {
            mask[i] = 1;
        }
    } else {
        for (size_t i = 0; i < dim->Size(); i++) {
            int64_t index = MakeWrapDim((*dim)[i], static_cast<int64_t>(selfShape.GetDimNum()));
            if (index < 0 || index >= static_cast<int64_t>(maxMaskLen) || mask[index] == 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ReduceMean dim value[%ld] invalid or repeat.", (*dim)[i]);
                return false;
            }
            mask[index] = 1;
        }
    }
    for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
        if (mask[i] == 0) {
            outShape.AppendDim(selfShape.GetDim(i));
        } else if (keepdim) {
            outShape.AppendDim(1);
        }
    }
    return true;
}

static const aclTensor* GenerateDimTensor(const aclTensor* self, const aclIntArray* dim, aclOpExecutor* executor)
{
    if (dim->Size() != 0) {
        return executor->ConvertToTensor(dim, DataType::DT_INT64);
    }
    FVector<int64_t> dimVector;
    for (size_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
        dimVector.emplace_back(i);
    }
    return executor->ConvertToTensor(dimVector.data(), dimVector.size(), DataType::DT_INT64);
}

static aclTensor* AllocAndInferReduceStdV2Update(const aclTensor* self, const aclTensor* mean, const aclIntArray* dim,
    bool ifStd, bool unbiased, bool keepdim, aclOpExecutor* executor)
{
    auto out = AllocReduceOut(self, executor);
    CHECK_RET(out != nullptr, nullptr);
    INFER_SHAPE(ReduceStdV2Update, OP_INPUT(self, mean), OP_OUTPUT(out), OP_ATTR(dim, ifStd, unbiased, keepdim));
    if (keepdim) {
        SetKeepDimShape(self, dim, out);
    }
    return out;
}

static aclTensor* AllocAndInferReduceStdV2UpdateCorrection(const aclTensor* self, const aclTensor* mean,
    const aclIntArray* dim, bool ifStd, bool unbiased, bool keepdim, int64_t correction, aclOpExecutor* executor)
{
    auto out = AllocReduceOut(self, executor);
    CHECK_RET(out != nullptr, nullptr);
    INFER_SHAPE(ReduceStdV2Update, OP_INPUT(self, mean), OP_OUTPUT(out),
        OP_ATTR(dim, ifStd, unbiased, keepdim, correction));
    if (keepdim) {
        SetKeepDimShape(self, dim, out);
    }
    return out;
}

const std::tuple<const aclTensor *, const aclTensor *> ReduceStdV2(const aclTensor* self, const aclIntArray* dim,
    int64_t correction, bool keepdim, bool isMeanOut, aclOpExecutor* executor)
{
    L0_DFX(ReduceStdV2, self, dim, correction, keepdim, isMeanOut);

    aclTensor *reduceStdOut = AllocReduceOut(self, executor);
    CHECK_RET(reduceStdOut != nullptr, std::tuple(nullptr, nullptr));

    aclTensor *reduceStdMeanOut = AllocReduceOut(self, executor);
    CHECK_RET(reduceStdMeanOut != nullptr, std::tuple(nullptr, nullptr));

    INFER_SHAPE(ReduceStdV2, OP_INPUT(self), OP_OUTPUT(reduceStdOut, reduceStdMeanOut),
                OP_ATTR(dim, correction, keepdim, isMeanOut));

    op::Shape outShape = self->GetViewShape();
    if (keepdim) {
        SetKeepDimShape(self, dim, reduceStdOut);
        outShape = reduceStdOut->GetViewShape();
        reduceStdOut->SetViewShape(outShape);
        reduceStdMeanOut->SetViewShape(outShape);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceStdV2, OP_INPUT(self), OP_OUTPUT(reduceStdOut, reduceStdMeanOut),
                                           OP_ATTR(dim, correction, keepdim, isMeanOut));
    if (ret !=  ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceStdV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }
    return std::tuple(reduceStdOut, reduceStdMeanOut);
}

const aclTensor* ReduceMean(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclOpExecutor* executor)
{
    L0_DFX(ReduceMean, self, dim, keepdim);
    op::Shape reduceShape;
    if (!InferReduceMeanShape(self->GetViewShape(), dim, keepdim, reduceShape)) {
        return nullptr;
    }
    if (self->GetViewShape().GetDimNum() == 0) {
        return self;
    }

    auto out = executor->AllocTensor(reduceShape, self->GetDataType());
    CHECK_RET(out != nullptr, nullptr);
    auto dimTensor = GenerateDimTensor(self, dim, executor);
    CHECK_RET(dimTensor != nullptr, nullptr);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceMean, OP_INPUT(self, dimTensor), OP_OUTPUT(out), OP_ATTR(keepdim, true));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceMean ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}

const aclTensor* ReduceStdWithMean(const aclTensor* self, const aclTensor* mean, const aclIntArray* dim,
    int64_t correction, bool keepdim, bool invert, float eps, aclOpExecutor* executor)
{
    L0_DFX(ReduceStdWithMean, self, mean, dim, correction, keepdim, invert, eps);
    auto out = AllocReduceOut(self, executor);
    CHECK_RET(out != nullptr, nullptr);

    bool unbiased = correction != 0;
    INFER_SHAPE(ReduceStdWithMean, OP_INPUT(self, mean), OP_OUTPUT(out),
        OP_ATTR(dim, unbiased, keepdim, invert, eps, correction));
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceStdWithMean, OP_INPUT(self, mean), OP_OUTPUT(out),
        OP_ATTR(dim, unbiased, keepdim, invert, eps, correction));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceStdWithMean ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}

const aclTensor* ReduceStdV2Update(const aclTensor* self, const aclTensor* mean, const aclIntArray* dim,
    bool unbiased, bool keepdim, aclOpExecutor* executor)
{
    L0_DFX(ReduceStdV2Update, self, mean, dim, unbiased, keepdim);
    bool ifStd = false;
    auto out = AllocAndInferReduceStdV2Update(self, mean, dim, ifStd, unbiased, keepdim, executor);
    CHECK_RET(out != nullptr, nullptr);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceStdV2Update, OP_INPUT(self, mean), OP_OUTPUT(out),
        OP_ATTR(dim, ifStd, unbiased, keepdim));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceStdV2Update ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}

const aclTensor* ReduceStdV2UpdateCorrection(const aclTensor* self, const aclTensor* mean, const aclIntArray* dim,
    int64_t correction, bool keepdim, aclOpExecutor* executor)
{
    L0_DFX(ReduceStdV2UpdateCorrection, self, mean, dim, correction, keepdim);
    bool ifStd = false;
    bool unbiased = correction != 0;
    auto out = AllocAndInferReduceStdV2UpdateCorrection(self, mean, dim, ifStd, unbiased, keepdim, correction, executor);
    CHECK_RET(out != nullptr, nullptr);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceStdV2Update, OP_INPUT(self, mean), OP_OUTPUT(out),
        OP_ATTR(dim, ifStd, unbiased, keepdim, correction));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceStdV2UpdateCorrection ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}
}  // namespace l0op
