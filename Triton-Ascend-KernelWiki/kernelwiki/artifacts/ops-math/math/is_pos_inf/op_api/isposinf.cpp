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
 * \file isposinf.cpp
 * \brief
 */

#include "isposinf.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(IsPosInf);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AiCore逻辑
static void IsPosInfAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(IsPosInfAiCore, self, out);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(IsPosInf, OP_INPUT(self), OP_OUTPUT(out));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IsPosInf ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
}

const aclTensor* IsPosInf(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(IsPosInf, self);
    // 判断走AiCore还是AiCPU，目前无AiCPU实现，默认走AiCore
    if (!IsAiCoreSupport(self)) {
        return nullptr;
    }
    // 根据输入shape申请输出tensor
    auto out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_BOOL);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "[isposinf] alloc out tensor failed.");
        return out;
    }
    IsPosInfAiCore(self, out, executor);
    return out;
}
} // namespace l0op
