/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conjugate_transpose_proto.h
 * \brief
 */
#ifndef OPS_OP_CONJUGATE_TRANSPOSE_PROTO_H_
#define OPS_OP_CONJUGATE_TRANSPOSE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Returns the complex conjugatetranspose.

* @par Inputs:
* @li x: A Tensor. Must be one of the following types: double, float32, float16, bfloat16, complex32, complex64,
complex128,
* int8, uint8, int16, uint16, int32, uint32, int64, uint64, qint8, quint8, qint16, quint16, qint32.
* @li perm: A Index. Must be one of the following types: int32, int64 \n
*
* @par Outputs:
* @li y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility.
* Compatible with tensorflow ConjugateTranspose operator.
*/
REG_OP(ConjugateTranspose)
    .INPUT(x, TensorType::BasicType())
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(ConjugateTranspose)
} // namespace ge
#endif
