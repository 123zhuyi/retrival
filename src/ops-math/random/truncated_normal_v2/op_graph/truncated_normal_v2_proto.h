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
 * \file truncated_normal_v2_proto.h
 * \brief
 */
#ifndef OP_PROTO_TRUNCATED_NORMAL_V2_H_
#define OP_PROTO_TRUNCATED_NORMAL_V2_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Outputs random values from a truncated normal distribution . \n

*@par Inputs:
*Inputs include:
*shape: A tensor. Must be one of the following types: int32, int64 . \n
*offset: A tensor. Must be int64 . The value is a multiple of 256 . \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0. If either `seed` or `seed2` 
are set to be non-zero, the random number generator is seeded by the given 
seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0. A second seed to avoid seed collision. \n
*@li dtype: An optional int.  Defaults to 0. The data type of y. It supports 1(float16), 27(bfloat16) and 0(float32).

*@par Outputs:
*@li y: A tensor of types: float16, float32, bfloat16 . A tensor of the specified shape
filled with random truncated normal values. \n
*@li offset: A tensor of types: int64 . A tensor is used to record the status and help generate new random numbers . \n

*@attention Constraints:
*If either seed or seed2 are set to be non-zero, the random number generator is seeded by the given seed. 
*Otherwise, it is seeded by a random seed.

*@par Third-party framework compatibility
Compatible with tensorflow TruncatedNormal operator.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/

REG_OP(TruncatedNormalV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({ DT_BF16, DT_FLOAT16, DT_FLOAT }))
    .OUTPUT(offset, TensorType({ DT_INT64 }))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(TruncatedNormalV2)

} // namespace ge

#endif // OP_PROTO_DIAG_V2_H_