/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "addr.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Addr);

// AICORE算子kernel
const aclTensor* Addr(
    const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2, const aclTensor* beta, const aclTensor* alpha,
    const op::DataType& hightDtype, aclOpExecutor* executor)
{
    L0_DFX(Addr, self, vec1, vec2, beta, alpha);
    op::Shape outerShape = {(vec1->GetViewShape())[0], (vec2->GetViewShape())[0]};
    auto output = executor->AllocTensor(outerShape, hightDtype);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Addr, OP_INPUT(self, vec1, vec2, beta, alpha), OP_OUTPUT(output));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Addr ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return output;
}
} // namespace l0op