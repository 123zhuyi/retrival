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
 * \file square_sum_v1_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SQUARE_SUM_V1_H_
#define OPS_OP_PROTO_INC_SQUARE_SUM_V1_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Confuse reducesumd and square.

* @par Inputs:
* x: A ND Tensor of type float16, float32, bfloat16. \n

* @par Attributes:
* Two attributes, including: \n
* @li axis: A required listint, specifies the dimensions to reduce.
* @li keep_dims: A optional bool, specifying whether to keep dimensions for the output Tensor. Defaults to "false". \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype as "x".

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(SquareSumV1)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SquareSumV1)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SQUARE_SUM_V1_H_
