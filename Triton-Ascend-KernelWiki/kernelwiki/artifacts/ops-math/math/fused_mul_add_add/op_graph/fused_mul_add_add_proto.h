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
 * \file fused_mul_add_add_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FUSED_MUL_ADD_ADD_H_
#define OPS_OP_PROTO_INC_FUSED_MUL_ADD_ADD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Fused multiply-add-add: y = x1 * x2 + x3 + x4, element-wise. x1 must carry
*        the full output shape; x2, x3 and x4 are broadcast up to x1 (NumPy rules).

* @par Inputs:
* Four inputs, including:
* @li x1: A ND tensor. Must be one of the following types: float16, float32, int32.
*     Its shape is the output shape.
* @li x2: A ND tensor. Must have the same dtype as x1; shape must be broadcastable up to x1.
* @li x3: A ND tensor. Must have the same dtype as x1; shape must be broadcastable up to x1.
* @li x4: A ND tensor. Must have the same dtype as x1; shape must be broadcastable up to x1. \n

* @par Outputs:
* y: A ND tensor. Has the same dtype and shape as x1. \n

* @attention Constraints:
* The runtime currently requires x1 to be the full output shape. Cases where x1 itself
* must be broadcast up (i.e. x1 is smaller than the output, e.g. x1=[1] with x3=[3,4])
* are not supported and fail at runtime.

* @par Third-party framework compatibility
* Compatible with the graph fusion of Mul followed by two Add operators
* (e.g. BatchMatmul + bias + residual patterns).
*/
REG_OP(FusedMulAddAdd)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(x3, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(x4, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(FusedMulAddAdd)

} // namespace ge

#endif // OPS_OP_PROTO_INC_FUSED_MUL_ADD_ADD_H_
