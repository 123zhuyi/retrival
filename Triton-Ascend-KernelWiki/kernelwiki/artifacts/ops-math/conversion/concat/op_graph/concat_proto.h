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
 * \file concat_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_CONCAT_PROTO_H_
#define OPS_OP_PROTO_INC_CONCAT_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* Two inputs, including:
* @li concat_dim: Must be one of the IndexNumberType: int32, int64.
* Specifies the dimension along which to concatenate .
* @li x: Dynamic input.A ND Tensor.
* Must be one of the BasicType:
  complex128, complex64, double, float32, float16, int16, int32, int64, int8,
  qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
  bfloat16, complex32, bool, HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0.. \n


* @par Attributes:
* N: An optional int8, int16, int32, or int64. Specifies the number of elements in "x" .
  Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "x" is a list of at least 2 "tensor" objects of the same type.
* @li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Concat. \n
*/
REG_OP(Concat)
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Concat)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CONCAT_PROTO_H_
