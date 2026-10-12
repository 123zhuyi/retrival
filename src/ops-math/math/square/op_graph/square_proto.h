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
 * \file elewise_calculation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Computes square of "x" element-wise.

*@par Inputs:
*One input:
* x: A ND Tensor. Must be one of the following types: float16, bfloat16, float32, float64, int32, int64, complex64,
*    complex128.

*@par Outputs:
*y: An ND or 5HD tensor. Support 1D ~ 8D. Shape and dtype of output, should be same shape and type as input.

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Square.
*/
REG_OP(Square)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Square)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
