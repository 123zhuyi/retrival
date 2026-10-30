/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "is_pos_inf.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(IsPosInf);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* IsPosInfAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(IsPosInfAiCore, self, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IsPosInf, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IsPosInf ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}

const aclTensor* IsPosInf(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(IsPosInf, self);
    if (!IsAiCoreSupport(self)) {
        return nullptr;
    }

    auto out = executor->AllocTensor(self->GetViewShape(), DataType::DT_BOOL, Format::FORMAT_ND);
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IsPosInf alloc output tensor failed."),
             return nullptr);
    const aclTensor* result = IsPosInfAiCore(self, out, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IsPosInfAiCore failed."),
             return nullptr);
    return result;
}
}
