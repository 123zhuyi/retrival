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
 * @file acos_grad_v2.h
 * @brief ACLNN L0 API 接口声明 - AcosGradV2 算子 (A2 / Ascend910B)
 *
 * 对齐 math/acos_grad README: 输入 y（前向 Acos 输入）、dy（上游梯度），输出 z（对原始输入的梯度）。
 * 公式: z = -dy / sqrt(1 - y^2)
 */

#ifndef OP_API_INC_LEVEL0_ACOS_GRAD_V2_H_
#define OP_API_INC_LEVEL0_ACOS_GRAD_V2_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* AcosGradV2(const aclTensor* y, const aclTensor* dy, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_ACOS_GRAD_V2_H_
