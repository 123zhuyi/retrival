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
 * \file reduce_sum_proto.h
 * \brief
 */

#ifndef REDUCE_SUM_PROTO_H_
#define REDUCE_SUM_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Computes the sum of elements across dimensions of a tensor.

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
* complex128, complex64, double, float32, float16, int16, int32, int64,
* int8, qint32, qint8, quint8, uint16, uint32, uint64, uint8, bfloat16,
* complex32.
* @li axes: A 1D list or tuple of IndexNumberType(int32 or int64).
* Specifies the dimensions to reduce.

* @par Attributes:
* keep_dims: An optional bool. If "true", retains reduced dimensions with
* length 1. Defaults to "false".
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x".

* @attention Constraints:
* @li The value range of "axes" is [-dims, dims - 1]. "dims"
  indicates the dimension length of "x".
* @li When converting ONNX to OM, if the axes of the ReduceSum operator is empty,
   it is recommended to use the sum function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sum.
*/
REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceSum)
} // namespace ge

#endif