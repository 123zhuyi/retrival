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
 * @file asin_with_agent.cpp
 * @brief ACLNN L0 API 实现 - AsinWithAgent 算子
 *
 * L0 API 职责：形状推导、dtype 支持检查、Kernel 调度
 *
 * 输出 dtype 规则：
 *   - 输入 FLOAT/FLOAT16/DOUBLE -> 输出与输入相同 dtype
 *   - 输入 INT8/INT16/INT32/INT64/UINT8/BOOL -> 输出 FLOAT32
 *
 * 迭代二：激活全部 9 种 dtype（TilingKey 0-8）
 *
 * 注意：DOUBLE 类型在 aclnn_asin_with_agent.cpp（L2 API）层已完成 Host 端 fp64->fp32 转换，
 *       L0 API 接收到的已是 fp32 tensor，此处仍注册 DOUBLE 支持以保持接口一致性。
 */

#include "asin_with_agent.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AsinWithAgent);

// 迭代二：激活全部 9 种 dtype
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,   // TilingKey=0（Group A fp32）
    DataType::DT_FLOAT16, // TilingKey=1（Group A fp16）
    DataType::DT_DOUBLE,  // TilingKey=2（Group B DOUBLE，L2层 Host 端转换）
    DataType::DT_INT8,    // TilingKey=3（Group C INT8）
    DataType::DT_INT16,   // TilingKey=4（Group C INT16）
    DataType::DT_INT32,   // TilingKey=5（Group C INT32）
    DataType::DT_INT64,   // TilingKey=6（Group C INT64）
    DataType::DT_UINT8,   // TilingKey=7（Group C UINT8）
    DataType::DT_BOOL,    // TilingKey=8（Group C BOOL）
};

// 判断输入 dtype 的输出 dtype
static op::DataType GetOutputDtype(op::DataType inputDtype)
{
    switch (inputDtype) {
        case DataType::DT_FLOAT:
            return DataType::DT_FLOAT;
        case DataType::DT_FLOAT16:
            return DataType::DT_FLOAT16;
        case DataType::DT_DOUBLE:
            return DataType::DT_DOUBLE;
        // 整数/BOOL 类型输出 FLOAT32
        case DataType::DT_INT8:
        case DataType::DT_INT16:
        case DataType::DT_INT32:
        case DataType::DT_INT64:
        case DataType::DT_UINT8:
        case DataType::DT_BOOL:
            return DataType::DT_FLOAT;
        default:
            return DataType::DT_FLOAT;
    }
}

static bool IsAiCoreSupport(const aclTensor* x) { return CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST); }

static bool AsinWithAgentInferShape(const op::Shape& xShape, op::Shape& outShape)
{
    outShape = xShape;
    return true;
}

static const aclTensor* AsinWithAgentAiCore(const aclTensor* x, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AsinWithAgentAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AsinWithAgent, OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AsinWithAgentAiCore failed."), return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 * 1. InferShape      - 形状推导
 * 2. IsAiCoreSupport - 判断执行路径
 * 3. AllocTensor     - 分配输出 Tensor（dtype 根据输入 dtype 确定）
 * 4. AsinWithAgentAiCore - 调用 Kernel
 */
const aclTensor* AsinWithAgent(const aclTensor* x, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!AsinWithAgentInferShape(x->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AsinWithAgent not supported: dtype=%d.", static_cast<int>(x->GetDataType()));
        return nullptr;
    }

    // 根据输入 dtype 确定输出 dtype
    op::DataType outDtype = GetOutputDtype(x->GetDataType());
    out = executor->AllocTensor(outShape, outDtype);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor for output failed.");
        return nullptr;
    }

    return AsinWithAgentAiCore(x, out, executor);
}

} // namespace l0op
