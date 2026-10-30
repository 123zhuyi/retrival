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
 * \file axpy_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_AXPY_H_
#define OPS_OP_PROTO_INC_AXPY_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Add tensor with scale y = x1 + x2*alpha. Support broadcasting operations.

* @par Inputs:
* @li x1: A ND Tensor dtype of int32, float16, float32, bfloat16.
* @li x2: A ND Tensor dtype of int32, float16, float32, bfloat16. \n

* @par Attributes:
* alpha: a float required attr, apply to x2:x2*alpha

* @par Outputs:
* y: A ND Tensor. should be broadcast shape of x1 and x2. \n

* @par Third-party framework compatibility:
* Compatible with the PyTorch operator Axpy.
*/
REG_OP(Axpy)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(alpha, Float)
    .OP_END_FACTORY_REG(Axpy)

} // namespace ge

#endif // OPS_OP_PROTO_INC_AXPY_H_
