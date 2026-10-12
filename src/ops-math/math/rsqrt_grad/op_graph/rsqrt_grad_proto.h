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
 * \file rsqrt_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_RSQRT_GRAD_H_
#define OPS_OP_PROTO_INC_RSQRT_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the backpropagation of the square root operation.

* @par Inputs:
* Two inputs, including:
* @li y: An NCHW, NHWC, ND tensor. Support 1D ~ 8D. Must be one of the following types:
 * float, int32, int8, double, complex64, complex128, float16, bfloat16.
* @li dy: A ND tensor of the same dtype, shape and format as "y".

* @par Outputs:
* z: A ND tensor of the same dtype, shape and format as "y".

* @see Matmul() | Rsqrt ()

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator RsqrtGrad.
*/
REG_OP(RsqrtGrad)
    .INPUT(y, TensorType({UnaryDataType, DT_INT32, DT_INT8, DT_BF16}))
    .INPUT(dy, TensorType({UnaryDataType, DT_INT32, DT_INT8, DT_BF16}))
    .OUTPUT(z, TensorType({UnaryDataType, DT_INT32, DT_INT8, DT_BF16}))
    .OP_END_FACTORY_REG(RsqrtGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_RSQRT_GRAD_H_

