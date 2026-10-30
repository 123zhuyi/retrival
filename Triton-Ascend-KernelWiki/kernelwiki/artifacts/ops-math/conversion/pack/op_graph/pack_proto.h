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
 * \file pack_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PACK_PROTO_OPS_H_
#define OPS_OP_PROTO_INC_PACK_PROTO_OPS_H_
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Packs the list of tensors in values into a tensor with rank one higher
* than each tensor in values, by packing them along the axis dimension.
* Given a list of length N of tensors of shape (A, B, C); if axis == 0 then
* the output tensor will have the shape (N, A, B, C) .

*@par Inputs:
* x: A list of N Tensors. Must be one of the following types: complex128,
* complex64, double, float32, float16, int16, int32, int64, int8, qint16,
* qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8, bfloat16,
* complex32. It's a dynamic input.

*@par Attributes:
*@li axis: An optional int, default value is 0.
*     Dimension along which to pack. The range is [-(R+1), R+1).
*@li N: An optional int, default value is 1. Number of tensors.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Pack.
*/
REG_OP(Pack)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(axis, Int, 0)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Pack)

} // namespace ge

#endif // OPS_OP_PROTO_INC_PACK_PROTO_OPS_H_
