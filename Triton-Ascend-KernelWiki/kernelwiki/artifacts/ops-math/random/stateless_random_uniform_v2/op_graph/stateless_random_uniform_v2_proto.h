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
 * \file stateless_random_uniform_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_UNIFORM_V2_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_UNIFORM_V2_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Outputs deterministic pseudorandom random integers from a uniform distribution. \n

* @par Inputs:
* @li shape: 1-D or empty tensor. The shape of the output tensor. Must be one of the following types: int32, int64.
* @li key: shape[1]. Key for the counter-based RNG algorithm. Must be one of the following types: uint64.
* @li counter: 1-D. Initial counter for the counter-based RNG algorithm. Must be one of the following types: uint64.
* @li alg: 0-D. The RNG(random number generator) algorithm. Must be one of the following types: int32. \n

* @par Attributes:
* dtype:Output data type. Must be one of the following types: float16, bfloat16, float32, double.
* Defaults to float32. \n

* @par Outputs:
* y: Returns Random values with specified shape.
* Must be one of the following types: float16, bfloat16, float32, double. \n

* @par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomUniformV2 operator.
*/

REG_OP(StatelessRandomUniformV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(key, TensorType({DT_UINT64}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .INPUT(alg, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_DOUBLE}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessRandomUniformV2)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_
