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
 * \file slice_proto.h
 * \brief
 */
#ifndef OP_PROTO_SLICE_H_
#define OP_PROTO_SLICE_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
* @brief Extracts a slice from a tensor.
*       This operation extracts a slice of size "size" from a tensor "x"
*       starting at the location specified by "offsets".

* @par Inputs:
* @li x: A Tensor. Must be one of the following types:
* bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, hifloat8, float8_e5m2, float8_e4m3fn
* float8_e8m0, float4_e2m1, float4_e1m2.
* @li offsets: A Tensor of type int32 or int64. The starting location for the slice.
* @li size: A Tensor of type int32 or int64. The tensor size for the slice. \n

* @attention Constraints:
* @li 0 <= offset[i] <= offset[i] + size[i] <= x_dim[i] for i in [0,n],
* n is the dimension of the tensor "x". \n
* @li offsets, size and x must have the same rank.

* @par Outputs:
* y: A Tensor. Has the same type as "x". The slice extracted from the tensor. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Slice.
*/
REG_OP(Slice)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(offsets, TensorType::IndexNumberType())
    .INPUT(size, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OP_END_FACTORY_REG(Slice)

} // namespace ge
#endif // OP_PROTO_SLICE_H_