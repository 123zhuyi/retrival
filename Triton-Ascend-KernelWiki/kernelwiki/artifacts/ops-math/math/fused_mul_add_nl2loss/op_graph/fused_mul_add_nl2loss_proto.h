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
 * \file fused_mul_add_nl2loss_proto.h
 * \brief FusedMulAddNL2loss proto 定义（与 built-in 910b 语义一致）
 */
#ifndef OPS_OP_PROTO_INC_FUSED_MUL_ADD_NL2LOSS_H_
#define OPS_OP_PROTO_INC_FUSED_MUL_ADD_NL2LOSS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes y1 = x1 * x3 + x2 and y2 = sum(x1^2 / 2).
*
* 注册体与 canndev built-in 定义逐字一致（ops_proto_legacy.h /
* elewise_calculation_ops.h 中的 REG_OP(FusedMulAddNL2loss)），
* 声明层 NumberType 宽约束；实际支持 fp16/fp32，由 def/tiling/kernel 侧把关
* （与 910b 相同：proto 宽声明、impl 层拦截）。

* @par Inputs:
* Three inputs, including:
* @li x1: An ND tensor (weight). Must be one of the following types: float16, float32. \n
* @li x2: An ND tensor (weight_grad). Has the same shape and dtype as "x1". \n
* @li x3: A scalar tensor (const_input), broadcast to the shape of "x1". Has the same dtype as "x1". \n

* @par Outputs:
* @li y1: An ND tensor. Has the same shape and dtype as "x1". \n
* @li y2: A scalar tensor, the L2 loss of "x1". Has the same dtype as "x1". \n

* @par Third-party framework compatibility
* Compatible with the fused Mul + AddN + L2Loss subgraph.
*/
REG_OP(FusedMulAddNL2loss)
    .INPUT(x1, TensorType::NumberType())
    .INPUT(x2, TensorType::NumberType())
    .INPUT(x3, TensorType::NumberType())
    .OUTPUT(y1, TensorType::NumberType())
    .OUTPUT(y2, TensorType::NumberType())
    .OP_END_FACTORY_REG(FusedMulAddNL2loss)

} // namespace ge

#endif // OPS_OP_PROTO_INC_FUSED_MUL_ADD_NL2LOSS_H_
