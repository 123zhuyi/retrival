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
 * @file fused_mul_add_n.h
 * @brief ACLNN L0 API 接口声明 - FusedMulAddN (y = x1 * x3[0] + x2)
 *
 * L0 API 是 ACLNN 底层实现，由 L2 API 调用。
 * 职责：形状推导（INFER_SHAPE）、Kernel 调度（ADD_TO_LAUNCHER_LIST_AICORE 调度到
 *       已注册的 FusedMulAddN aicore 算子）。
 */

#ifndef OP_API_INC_LEVEL0_OP_FUSED_MUL_ADD_N_OP_H_
#define OP_API_INC_LEVEL0_OP_FUSED_MUL_ADD_N_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

// y = x1 * x3[0] + x2。x1/x2/y 同 dtype、同 shape；x3 为同 dtype 的单元素标量张量（仅取 x3[0]）。
const aclTensor* FusedMulAddN(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_FUSED_MUL_ADD_N_OP_H_
