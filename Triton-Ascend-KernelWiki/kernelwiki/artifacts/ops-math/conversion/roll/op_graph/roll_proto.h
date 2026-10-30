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
 * \file roll_proto.h
 * \brief
 */
#ifndef ROLL_PROTO_H
#define ROLL_PROTO_H

namespace ge {
/**
*@brief Roll the tensor along the given dimension(s).
* Elements that are shifted beyond the last position are re-introduced at the first position.
* If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.

*@par Inputs:
*One input, including:
* x: A tensor. Must be one of the following types:
*     float16, bfloat16, float32, int32, uint32, int8, uint8, int64. \n

*@par Attributes:
* @li shifts: A required listInt. The number of places by which the elements of the tensor are shifted. 
*      The length of the array must be equal to dims, but if dims is empty, the array must be one-dimensional. \n
* @li dims: An optional listInt. Defaults to "None". Axis along which to roll. The value of the range is [0, number of dimensions for input -1] or [-number of dimensions for self, -1].


*@par Outputs:
* y: A Tensor with the same type and shape of x. \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator Roll. \n
*/
REG_OP(Roll)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8, DT_BF16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8, DT_BF16, DT_INT64}))
    .REQUIRED_ATTR(shifts, ListInt)
    .ATTR(dims, ListInt, {})
    .OP_END_FACTORY_REG(Roll)
} // namespace ge
// namespace ge
#endif // ROLL_PROTO_H
