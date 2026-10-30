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
 * \file stateless_random_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Outputs deterministic pseudorandom random integers from a uniform distribution. \n

* @par Inputs:
* @li shape: 1-D or empty tensor. The shape of the output tensor. Must be one of the following types: int64.
* @li seed: 0-D. seed for the counter-based RNG algorithm. Must be one of the following types: int64.
* @li offset: 0-D. offset for the counter-based RNG algorithm. Must be one of the following types: int64.
* @li from: 0-D scalar. Lower bound of the random range (inclusive). Must be one of the following types: int64.
* @li to: 0-D scalar. Upper bound of the random range (exclusive). Must be one of the following types: int64. \n

* @par Attributes:
* dtype:Output data type. Must be one of the following types: float16, bfloat16, float32, int64, int32,
* int16, int8, uint8, bool. Defaults to int32. \n

* @par Outputs:
* y: Returns Random values with specified shape.
* Must be one of the following types: float16, bfloat16, float32, int64, int32, int16, int8, uint8, bool. \n

* @par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandom operator.
*/

REG_OP(StatelessRandom)
    .INPUT(shape, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(from, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(to, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8, DT_BOOL}))
    .ATTR(dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(StatelessRandom)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_H_
