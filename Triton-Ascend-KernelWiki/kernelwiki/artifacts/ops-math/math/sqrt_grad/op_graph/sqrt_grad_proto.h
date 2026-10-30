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
 * \file sqrt_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SQRT_GRAD_H_
#define OPS_OP_PROTO_INC_SQRT_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the gradient of the square root of "x" with regard to its
* input. grad = dy * 0.5/y, where y = sqrt(x), and "dy" is the corresponding
* input gradient. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li y: A ND Tensor of type of UnaryDataType. Must be one of the following types: float64,
* float16, float32, complex128, complex64.
* @li dy: A ND Tensor. Has the same dtype as "y". \n

* @par Outputs:
* z: A ND Tensor. Has the same dtype as "y". \n

* @attention Constraints:
* "dy" has the same shape and type as "y".
*/
REG_OP(SqrtGrad)
    .INPUT(y, TensorType(UnaryDataType))
    .INPUT(dy, TensorType(UnaryDataType))
    .OUTPUT(z, TensorType(UnaryDataType))
    .OP_END_FACTORY_REG(SqrtGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SQRT_GRAD_H_
