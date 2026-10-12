/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATRIX_SET_DIAG_PROTO_H_
#define MATRIX_SET_DIAG_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Returns a batched matrix tensor with new batched diagonal values . \n

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. 2-8 dimensions. Must be one of the following types:
*    float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*    qint8, quint8, qint32, uint16, complex128, uint32, uint64, bfloat16, bool.
* @li diagonal: A Tensor of the same type as "x". 1-7 dimensions (must be one less than x).
     The last dimension is the smaller of the last two dimensions of x, 
     while the other dimensions are same as those in x. \n

* @par Outputs:
* y: A Tensor. Has the same type and shape as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MatrixSetDiag.
*/
REG_OP(MatrixSetDiag)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiag)
} // namespace ge

#endif
