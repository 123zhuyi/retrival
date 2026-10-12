/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_ASINH_H_
#define OPS_OP_PROTO_INC_ASINH_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Computes inverse hyperbolic sine of x element-wise.
* Given an input tensor, this function computes inverse hyperbolic sine for every element in the tensor.

*
*@par Inputs:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32, float64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same dtype as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Asinh.
*
*/
REG_OP(Asinh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Asinh)

}  // namespace ge
#endif  // OPS_OP_PROTO_INC_ASINH_H_
