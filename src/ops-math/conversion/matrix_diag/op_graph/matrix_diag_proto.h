/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATRIX_DIAG_PROTO_H
#define MATRIX_DIAG_PROTO_H

namespace ge {
/**
* @brief Returns a batched diagonal tensor with a given batched diagonal values . \n

* @par Inputs:
* x: A Tensor. Must be one of the following types:
*   double, float32, float16, bfloat16, complex32, complex64, complex128,
*   int8, uint8, int16, uint16, int32, uint32, int64, uint64, qint8, quint8, qint16, quint16, qint32, 
*   hifloat8, float8_e5m2, float8_e4m3fn, bool. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

* @attention Constraints:
* @li If x is an n-dimensional tensor, y has dimensions of n+1, and the first n dimensions of y are equal to the 
* first n dimensions of x, and the last dimension of y is consistent with the last dimension of x. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixDiag.
*/
REG_OP(MatrixDiag)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_BOOL}))
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_BOOL}))
    .OP_END_FACTORY_REG(MatrixDiag)
} // namespace ge

#endif