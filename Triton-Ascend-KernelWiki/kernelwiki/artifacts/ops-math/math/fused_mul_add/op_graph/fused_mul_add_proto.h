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
 * \file fused_mul_add_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FUSED_MUL_ADD_H_
#define OPS_OP_PROTO_INC_FUSED_MUL_ADD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Fused multiply-add: y = x1 * x2 + x3, element-wise with NumPy broadcasting.

* @par Inputs:
* Three inputs, including:
* @li x1: A ND tensor. Must be one of the following types: float16, float32, int32.
* @li x2: A ND tensor. Must have the same dtype as x1; shape must be broadcastable with x1.
* @li x3: A ND tensor. Must have the same dtype as x1; shape must be broadcastable with (x1 * x2). \n

* @par Outputs:
* y: A ND tensor. Has the same dtype as x1; shape is the broadcast shape of x1, x2 and x3. \n

* @par Third-party framework compatibility
* Compatible with TensorFlow/PyTorch graph fusion of Mul followed by Add.
*/
REG_OP(FusedMulAdd)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(x3, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(FusedMulAdd)

} // namespace ge

#endif // OPS_OP_PROTO_INC_FUSED_MUL_ADD_H_
