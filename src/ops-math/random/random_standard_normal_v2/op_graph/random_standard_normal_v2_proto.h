/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANDOM_STANDARD_NORMAL_V2_PROTO_H
#define RANDOM_STANDARD_NORMAL_V2_PROTO_H

namespace ge {
/**
*@brief Outputs random values from a standard normal distribution. \n

*@par Inputs:
*Inputs include:
*@li shape: A 1-D tensor. Must be one of the following types: int32, int64. The shape of the output tensor.
*@li offset: A 1-D tensor of types: int64. A tensor is used to record the status and help generate new random numbers. 
* The value is a multiple of 256. \n

*@par Attributes:
*@li dtype: A required int. Defaults to 0. The data type of y. It supports 1(float16), 27(bfloat16) and 0(float32).
*@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero, 
* the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
*@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

*@par Outputs:
*@li y: A tensor of type float32, float16, bfloat16.
*@li offset: A 1-D tensor of types: int64. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(RandomStandardNormalV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(offset, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomStandardNormalV2)
}

#endif