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
 * \file stateless_uniform_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_UNIFORM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_UNIFORM_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Outputs deterministic pseudorandom values from a uniform distribution in [from, to),
*        bit-exact with CUDA/PyTorch H20 GPU uniform_() implementation. \n

* @par Inputs:
* @li shape: 1-D tensor. The shape of the output tensor. Must be one of the following types: int64.
* @li seed: 0-D scalar. Philox algorithm seed. Must be one of the following types: int64.
* @li offset: 0-D scalar. Philox algorithm offset. Must be one of the following types: int64.
* @li from: 0-D scalar. Lower bound of the random range (inclusive). Must be one of the following types: double.
* @li to: 0-D scalar. Upper bound of the random range (exclusive). Must be one of the following types: double. \n

* @par Attributes:
* dtype: Output data type. Must be one of the following types: float16, bfloat16, float32.
* Defaults to float32. \n

* @par Outputs:
* y: Returns random values with specified shape. Values are in [from, to).
* Must be one of the following types: float16, bfloat16, float32. \n

* @par Third-party framework compatibility
* Compatible with PyTorch torch.Tensor.uniform_() operator (bit-exact with CUDA H20).
*/

REG_OP(StatelessUniform)
    .INPUT(shape, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .INPUT(from, TensorType({DT_DOUBLE}))
    .INPUT(to, TensorType({DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessUniform)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_STATELESS_UNIFORM_H_
