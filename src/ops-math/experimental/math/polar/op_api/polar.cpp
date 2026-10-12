/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file polar.cpp
 * \brief Polar L0 API：分配 complex64 输出张量并调度 AICore kernel。
 */

#include "polar.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Polar);

static const aclTensor* PolarAiCore(const aclTensor* input, const aclTensor* angle, const aclTensor* out,
                                    aclOpExecutor* executor)
{
    L0_DFX(PolarAiCore, input, angle, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Polar, OP_INPUT(input, angle), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PolarAiCore add to launcher failed."),
             return nullptr);
    return out;
}

const aclTensor* Polar(const aclTensor* input, const aclTensor* angle, aclOpExecutor* executor)
{
    // input 经 op_api 层 BroadcastTo 到 out shape；angle 可能保留原 [K]（inner-bcast，kernel 周期复用）。
    // out shape = input.shape（= 广播后输出形状），dtype = complex64。
    Shape outShape = input->GetViewShape();
    const aclTensor* out = executor->AllocTensor(outShape, op::DataType::DT_COMPLEX64);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Polar alloc output tensor failed."), return nullptr);
    return PolarAiCore(input, angle, out, executor);
}

} // namespace l0op
