/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANDOM_UNIFORM_INT_V2_PROTO_H
#define RANDOM_UNIFORM_INT_V2_PROTO_H

namespace ge {
/**
*@brief Outputs random integers from a uniform distribution with a given offset. \n

*@par Inputs:
*Inputs include:
*@li shape: A 1-D Tensor. Must be one of the following types: int32, int64. The shape of the output tensor.
*@li min: A 1-D Tensor. Must be one of the following types: int32, int64.
*@li max: A 1-D Tensor. Must be one of the following types: int32, int64.
*@li offset: A 1-D Tensor, should be const data. Must be one of the following types: int64.
The value of offset should not be less than 0 and will be set to the default value 0 if it is negative.  \n

*@par Attributes:
*@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero, 
the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

*@par Outputs:
*@li y: A Tensor. Has the same type as min. \n
*@li offset: A 1-D Tensor, should be const data. Must be one of the following types: int64. \n
*/
REG_OP(RandomUniformIntV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(min, TensorType({DT_INT32, DT_INT64}))
    .INPUT(max, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(offset, TensorType({DT_INT64}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniformIntV2)
}

#endif