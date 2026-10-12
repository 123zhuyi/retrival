/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_CONCAT_V2_H_
#define OPS_OP_PROTO_CONCAT_V2_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* Two inputs, including:
* @li Dynamic input "x" is A ND Tensor.
* Must be one of the following types: bfloat16, float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, uint16,
*     complex128, uint32, uint64, qint16, quint16, bool, string, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0.
* @li concat_dim: A 0D Tensor (scalar) with dtype int32, or int64. Specifies the dimension along which to concatenate .
\n

* @par Attributes:
* N: An optional int includes all types of int.
* Specifies the number of elements in "x". Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* "x" is a list of at least 2 "tensor" objects of the same type . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ConcatV2.
*/
REG_OP(ConcatV2)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2)
} // namespace ge

#endif // OPS_OP_PROTO_CONCAT_V2_H_
