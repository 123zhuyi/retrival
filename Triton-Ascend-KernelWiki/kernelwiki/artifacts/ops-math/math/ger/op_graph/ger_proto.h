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
 * \file ger_proto.h
 * \brief
*/

#ifndef OPS_OP_PROTO_INC_GER_H_
#define OPS_OP_PROTO_INC_GER_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes the outer product of two 1D vectors . \n

*@par Inputs:
*The input x1 and x2 has to be a 1D vector.Inputs include:
*@li x1:A Tensor. Must be one of the following types: float16, float32, bfloat16. 
Shape is [N] . \n
*@li x2:A Tensor. Must have the same type as x1. Shape is [M] . \n

*@par Outputs:
*y:A Tensor. Has the same type as x1. \n
*/

REG_OP(Ger)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(Ger)
} // namespace ge

#endif // OPS_OP_PROTO_INC_GER_H_