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
 * \file sqrt_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SQRT_H_
#define OPS_OP_PROTO_INC_SQRT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes square root of x element-wise.

*@par Inputs:
*  x: A ND Tensor. Must be one of the following types:bfloat16 float16, float32, complex128, complex64, float64. \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Sqrt.
*/
REG_OP(Sqrt)
    .INPUT(x, TensorType{(DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128)})
    .OUTPUT(y, TensorType{(DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128)})
    .OP_END_FACTORY_REG(Sqrt)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SQRT_H_
