/**
 * Copyright (c) 2025 Huawei Technologies Co.: Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS; WITHOUT WARRANTIES OF ANY KIND; EITHER EXPRESS OR IMPLIED;
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT; MERCHANTABILITY; OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CIRCULAR_PAD_BACKWARD_COMMON_H
#define CIRCULAR_PAD_BACKWARD_COMMON_H

#include "conversion/pad_v3_grad/op_api/padv3grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"

using namespace op;

static const string CIRCULAR_MODE = "circular";
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                     op::DataType::DT_BF16};

inline static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding,
                                const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 检查gradOutput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, dtypeSupportList, return false);

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查gradInput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, dtypeSupportList, return false);

    // gradOutput, self和gradInput数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradInput, self->GetDataType(), return false);
    return true;
}

inline static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    if (op::IsPrivateFormat(gradOutput->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW, NCL");
        return false;
    }
    OP_CHECK(
        gradOutput->GetViewFormat() == self->GetViewFormat() &&
            gradOutput->GetViewFormat() == gradInput->GetViewFormat(),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format of input and output should be equal, gradOutput [%s], self [%s], gradInput [%s].",
                op::ToString(gradOutput->GetViewFormat()).GetString(), op::ToString(self->GetViewFormat()).GetString(),
                op::ToString(gradInput->GetViewFormat()).GetString()),
        return false);
    return true;
}

typedef bool (*CheckShapeFn)(const aclTensor*, const aclTensor*, const aclIntArray*, const aclTensor*);

inline static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding,
                                      const aclTensor* gradInput, CheckShapeFn checkShapeFn,
                                      aclnnStatus nullPtrErrorCode)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, padding, gradInput), nullPtrErrorCode);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckFormat(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(checkShapeFn(gradOutput, self, padding, gradInput), ACLNN_ERR_PARAM_INVALID);

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

static aclnnStatus InputPreprocess(const aclTensor*& gradOutput, const aclTensor*& self, aclOpExecutor* executor)
{
    // 如果非连续，需要转连续
    gradOutput = l0op::Contiguous(gradOutput, executor);
    CHECK_RET(gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    self = l0op::Contiguous(self, executor);
    CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

#endif // CIRCULAR_PAD_BACKWARD_COMMON_H
