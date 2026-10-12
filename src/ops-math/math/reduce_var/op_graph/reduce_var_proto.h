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
 * \file reduce_var_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_REDUCE_VAR_H_
#define OPS_OP_PROTO_INC_REDUCE_VAR_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Calculates the variance and average value of tensors.

* @par Inputs:
* x: A tensor. Format supports ND. Must be one of the following types: float32, float16, bfloat16. \n

* @par Attributes:
* Four Attributes, including:
* @li dim: The dimensions to reduce. An optional listint, Defaults to "None".
*     If None (the default), reduces all dimensions.
*     Must be in the range [-rank(x), rank(x)).

* @li correction: An optional int. Used for Bessel's correction. Defaults to 1.

* @li keepdim: An optional bool. Defaults to "False".
*     If "True", Keep the original tensor dimension.
*     If "False", Do not keep the original tensor dimension.

* @li is_mean_out: An optional bool. Defaults to "True".
*     If "True", Output the mean.
*     If "False", Do not output the mean. \n

* @par Outputs:
* Two Outputs, including:
* @li var: A tensor, the variance of x. Has the same type and format as "x".
* @li mean: A tensor, the mean of x. Has the same type and format as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator var and var_mean.
*/
REG_OP(ReduceVar)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, ListInt, {})
    .ATTR(correction, Int, 1)
    .ATTR(keepdim, Bool, false)
    .ATTR(is_mean_out, Bool, true)
    .OP_END_FACTORY_REG(ReduceVar)

} // namespace ge

#endif // OPS_OP_PROTO_INC_REDUCE_VAR_H_

