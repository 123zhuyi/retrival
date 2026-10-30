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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "add_mat_mat_elements.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AddMatMatElements);

// Ascend950（DAV_3510）支持的 dtype
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* a)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return CheckType(a->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    }
    // 其他架构暂不支持
    return false;
}

static bool AddMatMatElementsInferShape(const op::Shape& aShape, op::Shape& outShape)
{
    // 输出 shape = 输入 shape（无广播）
    outShape = aShape;
    return true;
}

static const aclTensor* AddMatMatElementsAiCore(
    const aclTensor* a, const aclTensor* b, const aclTensor* c, float alpha, float beta, const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(AddMatMatElementsAiCore, a, b, c, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AddMatMatElements, OP_INPUT(a, b, c), OP_OUTPUT(out), OP_ATTR(alpha, beta));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddMatMatElementsAiCore failed."), return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 *   1. InferShape  - 形状推导（输出 shape = 输入 a 的 shape）
 *   2. IsAiCoreSupport - 判断执行路径
 *   3. AllocTensor - 分配输出 Tensor
 *   4. AiCore      - 调用 Kernel
 */
const aclTensor* AddMatMatElements(
    const aclTensor* a, const aclTensor* b, const aclTensor* c, float alpha, float beta, aclOpExecutor* executor)
{
    Shape outShape;
    if (!AddMatMatElementsInferShape(a->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AddMatMatElements: InferShape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(a)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AddMatMatElements: AiCore not supported for dtype=%d.",
            static_cast<int>(a->GetDataType()));
        return nullptr;
    }

    const aclTensor* out = executor->AllocTensor(outShape, a->GetDataType());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddMatMatElements: AllocTensor failed.");
        return nullptr;
    }

    return AddMatMatElementsAiCore(a, b, c, alpha, beta, out, executor);
}

} // namespace l0op
