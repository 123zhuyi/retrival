/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_CASE_CONDITION_PROTO_H_
#define OPS_MATH_CASE_CONDITION_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief x[0] is i, x[1] is j and x[2] is k when algorithm is LU,
y = 0 when i >= k && j < k,
y = 1 when i == k && j == k,
y = 2 when i > k && j == k,
y = 3 when i == k && j > k,
y = 4 when i > k && j > k,
default y = 5
use for lu decomposition
*@par Inputs:
*x: A Tensor of type int32/int64/uint64. \n

*@par Attributes:
*algorithm: A string, only support LU now.
*@par Outputs:
*y: A Tensor of type int32.
*/
REG_OP(CaseCondition)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(algorithm, String, "LU")
    .OP_END_FACTORY_REG(CaseCondition)

} // namespace ge

#endif // OPS_MATH_CASE_CONDITION_PROTO_H_
