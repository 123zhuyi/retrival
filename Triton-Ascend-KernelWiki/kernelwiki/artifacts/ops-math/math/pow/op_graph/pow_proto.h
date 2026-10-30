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
 * \file pow_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_POW_H_
#define OPS_OP_PROTO_INC_POW_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the power of "x1" to "x2". Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types:
*     bfloat16, float16, float32, int32, int64, int8, int16, uint8, double, complex64, complex128.
* @li x2: A ND Tensor of the same dtype as "x1". \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype as "x1". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Pow.
*/
REG_OP(Pow)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8, DT_INT16,
                              DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .DATATYPE(T2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8, DT_INT16,
                              DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Pow)

} // namespace ge

#endif // OPS_OP_PROTO_INC_POW_H_

