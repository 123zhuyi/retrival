/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file xlog1py_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_XLOG1PY_H_
#define OPS_OP_PROTO_INC_XLOG1PY_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Computes x * log1p(y) element-wise. Returns 0 if x == 0.

* @par Inputs:
* Two inputs, including:
* x: An ND Tensor. Must be one of: float16, float32, bfloat16, float64, complex64, complex128.
* y: An ND Tensor. Must be one of: float16, float32, bfloat16, float64, complex64, complex128.

* @par Outputs:
* z: An ND Tensor with broadcast shape of x and y. Has the same dtype as x and y.

* @par Third-party framework compatibility
* Compatible with the TensorFlow/PyTorch operator xlog1py.
*/
REG_OP(Xlog1py)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128}))
    .INPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128}))
    .OUTPUT(z, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Xlog1py)

} // namespace ge

#endif // OPS_OP_PROTO_INC_XLOG1PY_H_
