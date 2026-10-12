/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gcd.h"
#include "opdev/make_op_executor.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Gcd);

// AiCore支持的Gcd类型
static const std::initializer_list<op::DataType> GCD_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_INT32, op::DataType::DT_INT64,
};

static bool IsRegisteredSameDtypeSignature(const aclTensor* self, const aclTensor* other)
{
    if (self == nullptr || other == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Gcd L0 input must not be null.");
        return false;
    }
    const auto selfType = self->GetDataType();
    const auto otherType = other->GetDataType();
    if (!CheckType(selfType, GCD_AICORE_DTYPE_SUPPORT_LIST) || !CheckType(otherType, GCD_AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd L0 input dtypes %s and %s must be one of %s.",
                ToString(selfType).GetString(), ToString(otherType).GetString(),
                ToString(GCD_AICORE_DTYPE_SUPPORT_LIST).GetString());
        return false;
    }
    if (otherType != selfType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd L0 requires same-dtype inputs, but got self %s and other %s.",
                ToString(selfType).GetString(), ToString(otherType).GetString());
        return false;
    }
    return true;
}

static bool IsRegisteredKernelSignature(const aclTensor* self, const aclTensor* other, op::DataType outputType)
{
    if (self == nullptr || other == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Gcd L0 input must not be null.");
        return false;
    }
    const auto selfType = self->GetDataType();
    const auto otherType = other->GetDataType();
    if (!CheckType(selfType, GCD_AICORE_DTYPE_SUPPORT_LIST) || !CheckType(otherType, GCD_AICORE_DTYPE_SUPPORT_LIST) ||
        !CheckType(outputType, GCD_AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd L0 signature dtypes %s, %s and %s must be one of %s.",
                ToString(selfType).GetString(), ToString(otherType).GetString(), ToString(outputType).GetString(),
                ToString(GCD_AICORE_DTYPE_SUPPORT_LIST).GetString());
        return false;
    }
    if (otherType == selfType && outputType == selfType) {
        return true;
    }
    if (!gcd::IsRegisteredMixedKernelSignature(selfType, otherType, outputType)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd L0 has no registered kernel for dtype signature %s, %s -> %s.",
                ToString(selfType).GetString(), ToString(otherType).GetString(), ToString(outputType).GetString());
        return false;
    }
    return true;
}

// AiCore的执行逻辑
static void GcdAiCore(const aclTensor* self, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(GcdAiCore, self, other, out);

    ADD_TO_LAUNCHER_LIST_AICORE(Gcd, OP_INPUT(self, other), OP_OUTPUT(out));
}

const aclTensor* Gcd(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    L0_DFX(Gcd, self, other);

    // 目前Gcd无AiCPU,仅支持AiCore
    if (!IsRegisteredSameDtypeSignature(self, other)) {
        return nullptr;
    }

    // 对self和other的shape进行broadcast
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self tensor shape:%s and other tensor shape:%s can't broadcast.",
                ToString(self->GetViewShape()).GetString(), ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    aclTensor* gcdOut = executor->AllocTensor(broadcastShape, self->GetDataType());
    if (gcdOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "[Gcd] alloc out tensor failed.");
        return gcdOut;
    }
    GcdAiCore(self, other, gcdOut, executor);
    return gcdOut;
}

const aclTensor* GcdToOutput(const aclTensor* self, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(GcdToOutput, self, other, out);

    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GcdToOutput output must not be null.");
        return nullptr;
    }
    if (!IsRegisteredSameDtypeSignature(self, other)) {
        return nullptr;
    }
    if (out->GetDataType() != self->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GcdToOutput requires self, other and out to use the same dtype.");
        return nullptr;
    }

    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self tensor shape:%s and other tensor shape:%s can't broadcast.",
                ToString(self->GetViewShape()).GetString(), ToString(other->GetViewShape()).GetString());
        return nullptr;
    }
    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gcd output shape does not match inferred broadcast shape.");
        return nullptr;
    }

    GcdAiCore(self, other, out, executor);
    return out;
}

const aclTensor* GcdWithOutputType(const aclTensor* self, const aclTensor* other, op::DataType outputType,
                                   aclOpExecutor* executor)
{
    L0_DFX(GcdWithOutputType, self, other);

    if (!IsRegisteredKernelSignature(self, other, outputType)) {
        return nullptr;
    }

    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self tensor shape:%s and other tensor shape:%s can't broadcast.",
                ToString(self->GetViewShape()).GetString(), ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    aclTensor* gcdOut = executor->AllocTensor(broadcastShape, outputType);
    if (gcdOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "[GcdWithOutputType] alloc out tensor failed.");
        return gcdOut;
    }

    GcdAiCore(self, other, gcdOut, executor);
    return gcdOut;
}
} // namespace l0op
