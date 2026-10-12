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
 * \file tanh_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_TANH_GRAD_H_
#define OPS_OP_PROTO_INC_TANH_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes the gradient for the tanh of "x" .

*@par Inputs:
*Two inputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32, bfloat16,
*     double, complex64, complex128.
* @li dy: A Tensor. Must be one of the following types: float16, float32, bfloat16,
*     double, complex64, complex128. \n

*@par Attributes:
* @li complex_conj: An optional attribute indicates whether to use conjugate operations for complex dtype.

*@par Outputs:
*z: A Tensor. Must be one of the following types: float16, float32, bfloat16,
*     double, complex64, complex128.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator TanhGrad.
*/
REG_OP(TanhGrad)
    .INPUT(y, TensorType::UnaryDataType())
    .INPUT(dy, TensorType::UnaryDataType())
    .ATTR(complex_conj, Bool, false)
    .OUTPUT(z, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(TanhGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_TANH_GRAD_H_
