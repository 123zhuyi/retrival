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
 * \file mul_proto.h
 * \brief
 */

#ifndef MUL_PROTO_H_
#define MUL_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns x1 * x2 element-wise.
* y = x1 * x2. Support broadcasting operations.

* @par Inputs:
* @li x1: A ND tensor. Must be one of the following types: bool, float16, float32, bfloat16,
* float64, uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.
* @li x2: A ND tensor. Must be one of the following types: bool, float16, float32, bfloat16,
* float64, uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.
* The shape of x1 and x2 must meet the requirements of the broadcast relationship.

* @par Outputs:
* y: A ND tensor. Must be one of the following types: bool, float16, float32, float64, bfloat16,
* uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.

* @attention Constraints:
* "x1" and "x2" have incompatible shapes or types.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Multiply.
*/
REG_OP(Mul)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                              DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                              DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T2, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                              DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                              DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Mul)

}  // namespace ge

#endif  // MUL_PROTO_H_
