/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file asinh_with_agent.cpp
 * @brief ACLNN L0 API 实现 - AsinhWithAgent 一元算子
 *
 * L0 API 职责：形状推导、Kernel 调度
 * 注意：Kernel 侧只接受 float16 / float32。
 *      其他 dtype 由 L2 层（aclnn_asinh_with_agent.cpp）处理 Cast 后调用本接口。
 */

#include "asinh_with_agent.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AsinhWithAgent);

// Kernel 侧原生支持的 dtype（float16 / float32）
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_FLOAT};

static bool IsAiCoreSupport(const aclTensor* x) { return CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST); }

static bool AsinhWithAgentInferShape(const op::Shape& xShape, op::Shape& outShape)
{
    outShape = xShape;
    return true;
}

static const aclTensor* AsinhWithAgentAiCore(const aclTensor* x, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AsinhWithAgentAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AsinhWithAgent, OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AsinhWithAgentAiCore failed."), return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 * 1. InferShape      - 输出形状 = 输入形状
 * 2. IsAiCoreSupport - 判断执行路径（float16 / float32）
 * 3. AllocTensor     - 分配输出 Tensor
 * 4. AsinhWithAgentAiCore - 调用 Kernel
 */
const aclTensor* AsinhWithAgent(const aclTensor* x, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!AsinhWithAgentInferShape(x->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AsinhWithAgent: InferShape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(x)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "AsinhWithAgent: dtype %d not supported by AiCore. "
            "Expected float16 or float32.",
            static_cast<int>(x->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(outShape, x->GetDataType());

    return AsinhWithAgentAiCore(x, out, executor);
}

} // namespace l0op
