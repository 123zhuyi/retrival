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
 * \file fill_diagonal.cpp
 * \brief
 */

#include "fill_diagonal.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FillDiagonal);

// AICPU算子kernel
static const aclTensor* FillDiagonalAiCpu(
    const aclTensor* self, const aclScalar* fillValue, bool wrap, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(FillDiagonalAiCpu, self, fillValue, wrap, out);
    static internal::AicpuTaskSpace space("FillDiagonal");
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将FillDiagonal算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        FillDiagonal, OP_ATTR_NAMES({"fill_value", "wrap"}), OP_INPUT(self), OP_OUTPUT(out),
        OP_ATTR(fillValue->ToFloat(), wrap));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FillDiagonalAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* FillDiagonal(const aclTensor* self, const aclScalar* fillValue, bool wrap, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    CHECK_RET(out != nullptr, nullptr);
    return FillDiagonalAiCpu(self, fillValue, wrap, out, executor);
}
} // namespace l0op