/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gcd.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Gcd);

// AiCore支持的Gcd类型
static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT16, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8, op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_INT16, op::DataType::DT_INT64};

// 判断走AiCore还是AiCPU
static bool IsAiCoreSupport(const aclTensor* self, const aclTensor* other)
{
    std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = IsRegBase() ? ASCEND950_AICORE_DTYPE_SUPPORT_LIST :
                                                                                ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
    if (!CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    if (other->GetDataType() != self->GetDataType()) {
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
    if (!IsAiCoreSupport(self, other)) {
        return nullptr;
    }

    // 对self和other的shape进行broadcast
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self tensor shape:%s and other tensor shape:%s can't broadcast.",
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
} // namespace l0op
