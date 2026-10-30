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
 * \file stateless_randperm_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDPERM_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDPERM_PROTO_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Returns the random permutation of integers from 0 to n-1. \n

* @par Inputs:
* Inputs include:
* @li n: A 1-dimensional int64 tensor, shape must be [1].
* @li seed: A 1-dimensional int64 tensor, shape must be [1]. If seed is set to be -1, 
* and offset is set to be 0, the random number generator is seeded by a random seed. 
* Otherwise, it is seeded by the given seed.
* @li offset: A 1-dimensional int64 tensor, shape must be [1]. To avoid seed collision. \n

* @par Attributes:
* @li layout: An optional int. Defaults to 0.
* @li dtype: An optional type, used to specify the data type of output y. Defaults to int64. \n

* @par Outputs:
* @li y: A mutable tensor, shape is [n]. Must be one of the following types:
* float16, float32, double, int8, uint8, int16, int32, int64, bfloat16. \n

* @attention Constraints:
* The implementation for Randperm on Ascend uses AICPU, with bad performance.

* @par Third-party framework compatibility
* Compatible with Pytorch Randperm operator.
*/
REG_OP(StatelessRandperm)
    .INPUT(n, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT64, DT_INT32, DT_INT16,
        DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .ATTR(layout, Int, 0)
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(StatelessRandperm)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_
