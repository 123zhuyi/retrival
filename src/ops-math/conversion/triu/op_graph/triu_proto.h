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
 * \file triu_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_TRIU_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_TRIU_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input \n

* @par Inputs:
* x: A tensor, which supports 2-8 dimensions or be empty. Must be one of the following types:
* float16, bfloat16, float32, double, int32, uint8, int16, int8, complex64, int64,
* qint8, quint8, qint32, uint16, complex128, uint32, uint64, bool. \n

* @par Attributes:
* diagonal: An optional attribute indicates the diagonal to consider. Defaults to 0. \n

* @par Outputs:
* y: A tensor. Has the same type and shape as "x" . \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Triu.
*/
REG_OP(Triu)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL}))
    .ATTR(diagonal, Int, 0)
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))
    .OP_END_FACTORY_REG(Triu)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_TRIU_OPS_H_
