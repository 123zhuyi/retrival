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
 * @file fused_mul_add_n.cpp
 * @brief ACLNN L0 API 实现 - FusedMulAddN (y = x1 * x3[0] + x2)
 *
 * L0 API 职责：形状推导（INFER_SHAPE）、Kernel 调度（ADD_TO_LAUNCHER_LIST_AICORE）。
 * L2 API 职责：参数检查、Contiguous/ViewCopy 处理。
 *
 * 调度目标：op_host/fused_mul_add_n_def.cpp 注册的 FusedMulAddN aicore 算子
 *          （opFile.value="fused_mul_add_n"，仅 ascend910b / DAV_2201）。
 */

#include "fused_mul_add_n.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedMulAddN);

// FusedMulAddN 仅适配 Atlas A2（ascend910b / DAV_2201），支持 5 dtype（x1/x2/x3/y 同 dtype）。
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32, DataType::DT_INT16};

// 根据芯片类型、dtype 判断算子是否支持走 aicore。
static bool IsAiCoreSupport(const aclTensor* x1)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch != NpuArch::DAV_2201) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "FusedMulAddN only supports Atlas A2 (ascend910b/DAV_2201), but current npuArch=%d.",
            static_cast<int>(curArch));
        return false;
    }
    return CheckType(x1->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE 算子 kernel：将 FusedMulAddN 加入任务队列，调度到已注册的 aicore 算子。
static const aclTensor* FusedMulAddNAiCore(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(FusedMulAddNAiCore, x1, x2, x3, y);
    // x1/x2/x3/y dtype 必须完全一致（与 host tiling 强校验、proto 同 dtype 约束一致）。
    if (!(x1->GetDataType() == x2->GetDataType() && x1->GetDataType() == x3->GetDataType() &&
          x1->GetDataType() == y->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1, x2, x3, y dtype should all be same.");
        return nullptr;
    }
    // FusedMulAddN 是算子的 OpType，x1/x2/x3 是输入，y 是输出。
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedMulAddN, OP_INPUT(x1, x2, x3), OP_OUTPUT(y));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedMulAddNAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return y;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 * 1. AllocTensor     - 分配输出 Tensor（与 x1 同 dtype）
 * 2. INFER_SHAPE     - 形状推导（逐元素，y 与 x1 同 shape）
 * 3. IsAiCoreSupport - 判断执行路径（dtype + 芯片）
 * 4. AiCore          - 调用 Kernel
 */
const aclTensor* FusedMulAddN(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, aclOpExecutor* executor)
{
    L0_DFX(FusedMulAddN, x1, x2, x3);
    // 输出 y 与 x1 同 dtype；ND 格式，shape 由 INFER_SHAPE 推导。
    auto y = executor->AllocTensor(x1->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    CHECK_RET(y != nullptr, nullptr);

    auto ret = INFER_SHAPE(FusedMulAddN, OP_INPUT(x1, x2, x3), OP_OUTPUT(y), OP_EMPTY_ARG);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedMulAddN InferShape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(x1)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "x1 dtype %s should be in dtype support list [%s].",
            op::ToString(x1->GetDataType()).GetString(), op::ToString(AICORE_DTYPE_SUPPORT_LIST).GetString());
        return nullptr;
    }

    return FusedMulAddNAiCore(x1, x2, x3, y, executor);
}

} // namespace l0op
