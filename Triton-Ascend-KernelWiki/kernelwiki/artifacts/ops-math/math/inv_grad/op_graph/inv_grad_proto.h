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
 * \file inv_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_INV_GRAD_PROTO_H_
#define OPS_OP_INV_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge
{
/**
* @brief Computes "x" reciprocal grad, dx = -1*dy*y*y, where, "y = 1/x",
* and "dy" is the corresponding input gradient.  Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* int32, int8.
* @li grad: A ND Tensor. Has the same dtype as "x". \n

* @par Outputs:
* y: A ND Tensor, Has the same dtype as "x". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator InvGrad.
*/
REG_OP(InvGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OP_END_FACTORY_REG(InvGrad)
} // namespace ge
#endif