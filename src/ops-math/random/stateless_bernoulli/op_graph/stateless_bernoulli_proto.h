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
 * \file stateless_bernoulli_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_BERNOULLI_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_BERNOULLI_PROTO_H_

#include <vector>

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Generate bernoulli distribution for tensor input . \n

* @par Inputs:
include:
* @li shape: 1-D. The shape of the input tensor. A tensor of type int32, int64.
* @li prob: 0-D. A tensor of type float16, float32, double, bfloat16.
* Probability of bernoulli distribution, the value range from 0 to 1.
* @li seed: If seed is set to be -1, and offset is set to be 0, the random number
* generator is seeded by a random seed. Otherwise, it is seeded by the given seed.
* A tensor of type int64.
* @li offset: To avoid seed collision. A tensor of type int64, must be a multiple of 4.

* @par Attributes:
* dtype: The data type for the elements of the output tensor.

* @par Outputs:
* y: A tensor. The tensor of type support int8, uint8, int16, uint16, 
*  int32, uint32, int64, uint64, bool, float16, float, double, bf16. \n
*/
REG_OP(StatelessBernoulli)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64}))
    .INPUT(prob, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .INPUT(seed, TensorType({ DT_INT64 }))
    .INPUT(offset, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                            DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessBernoulli)

}   // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_
