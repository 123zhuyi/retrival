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
 * \file permute_proto.h
 * \brief
 */
#ifndef OP_PROTO_PERMUTE_PROTO_H_
#define OP_PROTO_PERMUTE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Permutes the dimensions according to order.
        The returned tensor's dimension i will correspond to the input dimension order[i] . \n

* @par Inputs:
* x: A ND tensor. Support 4D. Must be one of the following types: float16, float32 . \n

* @par Attributes:
* order: A permutation of the dimensions of "x".Type must be int32.Support any axis transformation.Defaults to "{0}"

* @par Outputs:
* y: A ND tensor. Support 4D. Has the same type as "x".

* @attention Constraints:
* The Attributes order must ensure that the provided dimensions are unique,do not repeat, and cover all dimensions of
"x". \n
*/
REG_OP(Permute)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(order, ListInt, {0})
    .OP_END_FACTORY_REG(Permute)

} // namespace ge
#endif // OP_PROTO_PERMUTE_PROTO_H_
