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
 * \file round_proto.h
 * \brief
 */
#ifndef OPS_OP_ROUND_PROTO_H_
#define OPS_OP_ROUND_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge
{
/**
*@brief Rounds the values of a tensor to the nearest integer, element-wise.
 * Rounds half to even.

*@par Inputs:
*Inputs including:
* x: An ND Tensor of type bfloat16, float16, float, int64, double, int32.

* @par Attributes:
* decimals: An optional int attr, number of decimal places to round to. Defaults to "0".

*@par Outputs:
*y: An ND Tensor. Has the same data type and shape as "x".
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Round.
*@attention Constraints:
* @li When the input value is between [-0.5, -0], the output value is 0.
* @li In the scenarios where the decimals is not zero:
*     The input data exceeds the range of (-347000, 347000), which may affect the precision errors.
*/
REG_OP(Round)
    .INPUT(x, TensorType(DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_INT64,
                         DT_DOUBLE))
    .OUTPUT(y, TensorType(DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_INT64,
                          DT_DOUBLE))
    .ATTR(decimals, Int, 0)
    .OP_END_FACTORY_REG(Round)
} // namespace ge
#endif