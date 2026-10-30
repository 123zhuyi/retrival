/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_max_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_REDUCE_MAX_H_
#define OPS_OP_PROTO_INC_REDUCE_MAX_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns the maximum of elements across dimensions of a Tensor .

* @par Inputs:
* Two inputs, including:
* @li x: A multi-dimensional Tensor of Must be the type of NumberType. Supported format list ["ND"]
* @li axes: A Scalar of type in IndexNumberType(IndexNumberType includes the
  following types: int32, int64.), specifying the axes information
  of the index with the maximum value. Supported format list ["ND"] \n

* @par Attributes:
* keep_dims: A bool, specifying whether to keep dimensions for the output Tensor.
* Optional and defaults to "false". \n
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

* @par Outputs:
* y: A multi-dimensional Tensor, specifying the maximum value of the
  corresponding axis in the tensor.
  Has the same type as "x". (If "keep_dims" is set to "false",
  the output dimensions are reduced by "dimension" compared with that of "x".
  Otherwise, the output has one fewer dimension than "x").Supported format list ["ND"]

* @attention Constraints:
* @li The value range of "axes" is [-dims, dims - 1]. "dims"
  indicates the dimension length of "x".
* @li When converting ONNX to OM, if the axes of the ReduceMax operator is empty,
   it is recommended to use the amax function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Max.
*/
REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMax)

} // namespace ge

#endif // OPS_OP_PROTO_INC_REDUCE_MAX_H_

