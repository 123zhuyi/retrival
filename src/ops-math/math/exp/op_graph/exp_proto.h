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
 * \file exp_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_EXP_H_
#define OPS_OP_PROTO_INC_EXP_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the exponential of "x" element-wise.

* @par Inputs:
* One input:
* x: A ND tensor. Must be one of the following types: bfloat16, float16, float32, double, complex64, complex128.
* Only when x's dtype is bfloat16, float16 or float32, attributes are valid and can be set.
* When x's dtype is double, complex64, complex128, attributes are invalid.

* @par Attributes:
* @li base: An optional attribute of type float32, specifying the base gamma. Must be positive or "-1.0", defaults to
"-1.0".
* @li scale: An optional attribute of type float32, specifying the scale alpha. Defaults to "1.0".
* @li shift: An optional attribute of type float32, specifying the shift beta. Defaults to "0.0".

* @par Outputs:
* y: A ND tensor of the same dtype as "x".

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Exp.
*/
REG_OP(Exp)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Exp)

} // namespace ge

#endif // OPS_OP_PROTO_INC_EXP_H_
