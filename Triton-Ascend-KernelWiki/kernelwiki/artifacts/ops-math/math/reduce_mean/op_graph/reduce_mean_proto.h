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
 * \file reduce_mean_proto.h
 * \brief
 */
#ifndef REDUCE_MEAN_PROTO_H_
#define REDUCE_MEAN_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Reduces "x" along the dimensions according to "axis".

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
*complex128, complex64, double, float32, float16, int64, int32, int16, int8,
*uint64, uint32, uint16, uint8, bfloat16. The data format supports ND.
* @li axes: The dimensions to reduce. Must be one of the following types:
* int, list, tuple, NoneType. Data type must be int32 or int64.
* If None (the default), reduces all dimensions.
* Must be in the range [-rank(x), rank(x)).

* @par Attributes:
* @li keep_dims: An optional bool. Defaults to false.
* If true, retains reduced dimensions with length 1.
* If false, the rank of the tensor is reduced by 1 for each entry in axis.
* @li noop_with_empty_axes: An optional bool. Defaults to true.
* If true, when axes = [], not reduce.
* If false, when axes = [], reduce all.
* @par Outputs:
* y: A tensor. Has the same type and format as "x".

* @attention Constraints:
* @li When converting ONNX to OM, if the axes of the ReduceMean operator is empty,
   and noop_with_empty_axes is true, it is recommended to use the mean function with dim explicitly
   set to all axes(e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator ReduceMean.
*/
REG_OP(ReduceMean)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMean)

} // namespace ge

#endif // OPS_OP_PROTO_INC_REDUCE_MEAN_H_

