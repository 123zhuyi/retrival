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
 * \file transpose_proto.h
 * \brief
 */
#ifndef OP_PROTO_TRANSPOSE_PROTO_H_
#define OP_PROTO_TRANSPOSE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Permutes the dimensions according to perm.
         The returned tensor's dimension i will correspond to the input dimension perm[i].

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types:
* bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bool, hifloat8, float8_e5m2,
* float8_e4m3fn, and the maximum dimension should not exceed 8 dimensions,
* and the shape should be consistent with output.
* @li perm: A Tensor of type int32 or int64. A permutation of the dimensions of "x", the value
* should be within the range of [0, number of dimensions for self -1].

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Transpose.
*/
REG_OP(Transpose)
    .INPUT(x, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                          DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                          DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                           DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                           DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                           DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Transpose)

} // namespace ge
#endif // OP_PROTO_TRANSPOSE_PROTO_H_
