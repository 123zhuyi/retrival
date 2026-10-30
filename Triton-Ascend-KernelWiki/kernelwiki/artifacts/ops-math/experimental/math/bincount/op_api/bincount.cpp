/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bincount.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Bincount);

static const aclTensor* BincountAiCore(const aclTensor* array, const aclTensor* size, const aclTensor* weights,
                                       const aclTensor* bins, aclOpExecutor* executor)
{
    L0_DFX(BincountAiCore, array, size, weights, bins);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Bincount, OP_INPUT(array, size, weights), OP_OUTPUT(bins));
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BincountAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return bins;
}

const aclTensor* Bincount(const aclTensor* array, const aclTensor* size, const aclTensor* weights,
                          const op::Shape& binsShape, op::DataType binsDtype, aclOpExecutor* executor)
{
    auto bins = executor->AllocTensor(binsShape, binsDtype, op::Format::FORMAT_ND);
    if (bins == nullptr) {
        return nullptr;
    }
    return BincountAiCore(array, size, weights, bins, executor);
}
} // namespace l0op
