/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bernoulli_mask.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(BernoulliMask);

aclTensor* BernoulliMask(const aclTensor* mask, aclTensor* out, bool maskAliasesOut, aclOpExecutor* executor)
{
    if (mask == nullptr || out == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BernoulliMask received a null argument.");
        return nullptr;
    }
    L0_DFX(BernoulliMask, mask, out);

    auto outputShape = op::ToShapeVector(out->GetViewShape());
    auto outputShapeArray = executor->AllocIntArray(outputShape.data(), outputShape.size());
    if (outputShapeArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BernoulliMask failed to allocate output_shape.");
        return nullptr;
    }

    const int64_t maskAliasMode = maskAliasesOut ? 1 : 0;
    auto args = op::GetOpArgContext(OP_INPUT(mask), OP_OUTPUT(out), OP_ATTR(outputShapeArray, maskAliasMode));
    if (args == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BernoulliMask failed to create its argument context.");
        return nullptr;
    }
    auto ret = CreatAiCoreKernelLauncher("BernoulliMask", BernoulliMaskOpTypeId(), executor, args);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BernoulliMask ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}

const aclTensor* BernoulliMask(const aclTensor* mask, const aclTensor* like, aclOpExecutor* executor)
{
    if (mask == nullptr || like == nullptr || executor == nullptr) {
        return nullptr;
    }
    auto out = executor->AllocTensor(like->GetViewShape(), like->GetDataType(), op::Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BernoulliMask failed to allocate output.");
        return nullptr;
    }
    return BernoulliMask(mask, out, false, executor);
}
} // namespace l0op
