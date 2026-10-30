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
 * \file concat_d_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_CONCAT_D_PROTO_H_
#define OPS_OP_PROTO_INC_CONCAT_D_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* One input:
* x: Dynamic input. A ND Tensor.
*    Must be one of the following types: bfloat16, float16, float32, int32,
*    int8, int16, int64, uint8, uint16, uint32, uint64, bool, double, complex64,
     HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E1M2、FLOAT4_E2M1. \n

* @par Attributes:
* @li concat_dim: A required int8, int16, int32, or int64.
                  Specifies the dimension along which to concatenate.
                  No default value.
* @li N:  An optional int8, int16, int32, or int64.
  Specifies the number of elements in "x". Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "x" is a list of at least 2 "tensor" objects of the same type.
* @li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Concat.
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Concat instead.
*/
REG_OP(ConcatD)
    .DYNAMIC_INPUT(x, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_BOOL, DT_INT32, DT_INT64,
                                  DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_COMPLEX64, DT_FLOAT8_E4M3FN,
                                  DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_BOOL, DT_INT32, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_COMPLEX64, DT_FLOAT8_E4M3FN,
                           DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .REQUIRED_ATTR(concat_dim, Int)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatD)
} // namespace ge

#endif // OPS_OP_PROTO_INC_CONCAT_D_PROTO_H_
