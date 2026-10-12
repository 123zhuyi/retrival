/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CIRCULAR_PAD_COMMON_H
#define CIRCULAR_PAD_COMMON_H

#include "conversion/pad_v3/op_api/padv3.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"

namespace op {

static const string kCircularMode = "circular";

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> kCircularPadDtypeSupportList = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT8};

inline static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, kCircularPadDtypeSupportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, kCircularPadDtypeSupportList, return false);

    // self和out数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

inline static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW, NCL");
        return false;
    }
    OP_CHECK(
        self->GetViewFormat() == out->GetViewFormat(),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], gradInoutput [%s].",
                op::ToString(self->GetViewFormat()).GetString(), op::ToString(out->GetViewFormat()).GetString()),
        return false);
    return true;
}

// 通过模板参数处理2d与3d的维度/padding差异
// 2d: MinDim=3, MaxDim=4, PaddingSize=4
// 3d: MinDim=4, MaxDim=5, PaddingSize=6
template <size_t MinDim, size_t MaxDim, size_t PaddingSize>
static bool CheckShape(const aclTensor* self, const aclIntArray* padding, const aclTensor* out)
{
    auto selfDimnum = self->GetViewShape().GetDimNum();
    OP_CHECK_MIN_DIM(self, MinDim, return false);
    OP_CHECK_MAX_DIM(self, MaxDim, return false);

    // self, out维度需要一致
    OP_CHECK(selfDimnum == out->GetViewShape().GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self, out dim should be same."), return false);

    // padding长度校验
    OP_CHECK(
        padding->Size() == PaddingSize,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be %zu, but got %zu.", PaddingSize, padding->Size()),
        return false);

    op::Shape expectShape;
    expectShape.SetDimNum(selfDimnum);
    size_t paddingDim = PaddingSize / 2;
    if (selfDimnum > paddingDim) {
        size_t dimToCompare = selfDimnum - paddingDim;
        for (size_t i = 0; i < dimToCompare; i++) {
            expectShape.SetDim(i, self->GetViewShape().GetDim(i));
        }
    }
    // 通用循环处理所有padding维度的shape计算
    for (size_t k = 0; k < paddingDim; k++) {
        expectShape.SetDim(selfDimnum - 1 - k,
                           self->GetViewShape().GetDim(selfDimnum - 1 - k) + (*padding)[k * 2] + (*padding)[k * 2 + 1]);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* padding, const aclTensor* out,
                                      bool (*checkShapeFn)(const aclTensor*, const aclIntArray*, const aclTensor*))
{
    // 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查shape是否满足约束
    CHECK_RET(checkShapeFn(self, padding, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetPaddingTensor(int64_t dim, const aclIntArray* padding, aclOpExecutor* executor)
{
    FVector<int64_t, op::MAX_DIM_NUM> paddingsVector;
    // 2 is the magnification
    for (size_t i = 2 * dim; i > 0; i -= 2) {
        if (i <= (size_t)padding->Size()) {
            // 2 and 1 indicate the element of padding is put into paddingsVector from the back to the front
            paddingsVector.emplace_back((*padding)[i - 2]);
            paddingsVector.emplace_back((*padding)[i - 1]);
        } else {
            paddingsVector.emplace_back(0);
            paddingsVector.emplace_back(0);
        }
    }
    // 2 is the magnification
    auto newpadding = executor->AllocIntArray(paddingsVector.data(), 2 * dim);
    auto paddingsTensor = executor->ConvertToTensor(newpadding, static_cast<op::DataType>(ACL_INT64));
    return paddingsTensor;
}

inline static aclnnStatus InputPreprocess(const aclTensor*& self, aclOpExecutor* executor)
{
    // 如果非连续，需要转连续
    self = l0op::Contiguous(self, executor);
    CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// 空tensor处理：通过minDim/maxDim参数区分2d(3D/4D)与3d(4D/5D)的校验差异
inline static aclnnStatus HandleEmptyTensor(const aclTensor* self, size_t minDim, size_t maxDim)
{
    if (self->GetViewShape().GetDimNum() == minDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected %zuD or %zuD tensor with possibly 0 batch size and other non-zero dimensions for input.",
                minDim, maxDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (self->GetViewShape().GetDimNum() == maxDim) {
        bool hasZeroDim = false;
        for (size_t i = 1; i < maxDim; i++) {
            if (self->GetViewShape().GetDim(i) == 0) {
                hasZeroDim = true;
                break;
            }
        }
        if (hasZeroDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected %zuD or %zuD tensor with possibly 0 batch size and other non-zero dimensions for input.",
                    minDim, maxDim);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

// 通用GetWorkspaceSize实现
// 2d: minDim=3, maxDim=4
// 3d: minDim=4, maxDim=5
template <size_t MinDim, size_t MaxDim, size_t PaddingSize>
inline static aclnnStatus CircularPadGetWorkspaceSizeImpl(const aclTensor* self, const aclIntArray* padding,
                                                          aclTensor* out, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor)
{
    CHECK_NOT_NULL(self, padding, out);
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, padding, out, CheckShape<MinDim, MaxDim, PaddingSize>);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        auto emptyRet = HandleEmptyTensor(self, MinDim, MaxDim);
        if (emptyRet != ACLNN_SUCCESS) {
            return emptyRet;
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用l0算子进行计算
    auto dim = self->GetViewShape().GetDimNum();
    ret = InputPreprocess(self, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    dim = self->GetViewShape().GetDimNum();
    auto paddingsTensor = GetPaddingTensor(dim, padding, uniqueExecutor.get());
    aclScalar* constantValueScalar = (uniqueExecutor.get())->AllocScalar(0);
    CHECK_RET(constantValueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto constantValueTensor = (uniqueExecutor.get())->ConvertToTensor(constantValueScalar, self->GetDataType());
    const aclTensor* padResult = nullptr;
    padResult = l0op::PadV3(self, paddingsTensor, constantValueTensor, kCircularMode, true, uniqueExecutor.get());
    CHECK_RET(padResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(padResult, out, uniqueExecutor.get());
    CHECK_RET((viewCopyResult != nullptr), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace op

#endif // CIRCULAR_PAD_COMMON_H
