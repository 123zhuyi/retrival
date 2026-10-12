/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "angle_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AngleV2);

static inline const aclTensor* AngleV2AiCore(const aclTensor* x, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AngleV2AiCore, x, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AngleV2, OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AngleV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

const aclTensor* AngleV2(const aclTensor* x, aclOpExecutor* executor)
{
    DataType outDtype = DataType::DT_FLOAT;
    if (x->GetDataType() == DataType::DT_FLOAT16) {
        outDtype = DataType::DT_FLOAT16;
    } else if (x->GetDataType() == DataType::DT_BF16) {
        outDtype = DataType::DT_BF16;
    }

    auto out = executor->AllocTensor(x->GetViewShape(), outDtype);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Alloc out tensor failed.");
        return nullptr;
    }
    return AngleV2AiCore(x, out, executor);
}
} // namespace l0op
