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
 * \file stateless_truncated_normal_v2_proto.h
 * \brief
 */
#ifndef OP_PROTO_STATELESS_TRUNCATED_NORMAL_V2_H_
#define OP_PROTO_STATELESS_TRUNCATED_NORMAL_V2_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Outputs random values from a truncated normal distribution (stateless version). \n

*@par Inputs:
*Inputs include:
*shape: A tensor. Must be one of the following types: int32, int64 . \n
*key: A tensor of type uint64 with shape [1]. The key for the random number generator. \n
*counter: A tensor of type uint64 with shape [2]. The counter for the random number generator. \n
*counter: Shape[1] for threefry, Shape[1] for philox. \n
*alg: A scalar tensor of type int32. The algorithm id (1 = Philox). \n
*alg: The default setting in this operator is 1. \n

*@par Attributes:
*@li dtype: An optional type. Defaults to DT_FLOAT. The data type of y. It supports 1(float16), 27(bfloat16) and 0(float32).

*@par Outputs:
*@li y: A tensor of types: float16, float32, bfloat16, double. A tensor of the specified shape
filled with random truncated normal values. \n

*@attention Constraints:
*This is a stateless version. The same key+counter input always produces the same output.
*Only Philox algorithm (alg=1) is currently supported.

*@par Third-party framework compatibility
Compatible with tensorflow StatelessTruncatedNormalV2 operator.
*/

REG_OP(StatelessTruncatedNormalV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(key, TensorType({DT_UINT64}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .INPUT(alg, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessTruncatedNormalV2)

} // namespace ge

#endif // OP_PROTO_STATELESS_TRUNCATED_NORMAL_V2_H_
