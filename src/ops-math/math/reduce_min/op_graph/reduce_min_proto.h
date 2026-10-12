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
 * \file reduce_min_proto.h
 * \brief
 */

#ifndef REDUCE_MIN_PROTO_H_
#define REDUCE_MIN_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Computes the minimum of elements across dimensions of a tensor .

* @par Inputs:
* @li x: A tensor. Must be the type of NumberType.
* @li axes: A tensor of type of IndexNumberType.(IndexNumberType
* includes: int32, int64.) Specifies the dimensions to reduce.
* Defaults to "None".

* @par Attributes:
* keep_dims: An optional bool. If "True", reduced dimensions will be retained.
* Defaults to "False".
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

* @par Outputs:
* y: A tensor. Must be the type of NumberType.

* @attention Constraints:
* @li If "axes = None", all dimensions will be reduced. "axes" must be in the
  range [-rank(input_shape), rank(input_shape)).
* @li When converting ONNX to OM, if the axes of the ReduceMin operator is empty,
   it is recommended to use the amin function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator reduce_min.
*/
REG_OP(ReduceMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMin)

} // namespace ge

#endif