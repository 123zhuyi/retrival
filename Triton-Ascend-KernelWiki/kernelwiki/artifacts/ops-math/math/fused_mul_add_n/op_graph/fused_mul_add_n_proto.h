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
 * \file fused_mul_add_n_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FUSED_MUL_ADD_N_H_
#define OPS_OP_PROTO_INC_FUSED_MUL_ADD_N_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Confuse broadcast, addn and mul.

* @par Inputs:
* Three inputs, including:
* @li x1: A ND tensor. Must be one of the following types:int32, int16,
* bfloat16, float16, float32. x1 is used to compute addn with x2. \n
* @li x2: A ND tensor of the same dtype and shape as "x1". x2 is used to compute addn with x1. \n
* @li x3: A ND tensor of the same dtype as "x1". The shape size should be 1.
* x3 is used to compute mul with the result of x1 + x2. \n

* @par Outputs:
* y: A ND tensor. Has the same dtype and shape as "x1". \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(FusedMulAddN)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .OP_END_FACTORY_REG(FusedMulAddN)

} // namespace ge

#endif // OPS_OP_PROTO_INC_FUSED_MUL_ADD_N_H_
