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
 * \file mul_addn_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MUL_ADDN_OPS_H_
#define OPS_OP_PROTO_INC_MUL_ADDN_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Confuse broadcast, addn and n mul operation. Support broadcasting operations.
 *@par Inputs:
*Two inputs, including:
* @li x1: A ND tensor. Must be one of the following types:bfloat16 float16, float32.
* @li x2: A ND tensor of the same dtype as "x1". \n

*@par Outputs:
*@ y: A ND tensor. Has the same dtype as "x1". \n

*@par Attributes:
* N:Represent the number of fusion mul operations, which is grater than or equal to 2.
    The support type is the int.

* @attention Constraints:
* @li The third dimension of x1 and the second dimension of x2 are both 1.
* @li The third dimension of x2 must be less than or equal to 2040.
*/

REG_OP(MulAddn)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(MulAddn)

} // namespace ge

#endif