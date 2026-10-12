/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_proto.h
 * \brief
 */
#ifndef OP_PROTO_ABS_PROTO_H_
#define OP_PROTO_ABS_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Computes the absolute value of a tensor.

* @par Inputs:
* One input, including:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* uint8, bool, bfloat16, float16, float32, double, int8, int16, int32, int64, complex32, complex64

* @par Outputs:
* y: A ND Tensor. Has the same dtype as "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Abs.
*/
REG_OP(Abs)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_UINT8, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16,
                             DT_INT32, DT_INT64, DT_COMPLEX32, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(Abs)

} // namespace ge
#endif // OP_PROTO_ABS_PROTO_H_