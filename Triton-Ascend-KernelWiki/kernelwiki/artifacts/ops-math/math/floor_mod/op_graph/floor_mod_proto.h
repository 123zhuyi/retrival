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
 * \file floor_mod_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FLOOR_MOD_H_
#define OPS_OP_PROTO_INC_FLOOR_MOD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns element-wise remainder of division.
* Consistent with: floor(x1/x2) * x2 + mod(x1, x2) = x1.
* Integer division by zero on NPU returns x1. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND tensor. Must be one of the following types:
*    int32, int64, float, float16, double, bfloat16
* @li x2: A ND tensor. Must have the same dtype as "x1".
*
* @par Outputs:
* y: A ND tensor. Has the same dtype as "x1".

* @attention Constraints:
* @li x2: The input data does not support 0
* @li When value of tensor exceeds 2048 , the accuracy of operator cannot guarantee the
* requirement of double thousandths in the mini platform
* @li Due to different architectures, the calculation results of this operator
* on NPU and CPU may be inconsistent
* @li If shape is expressed as (D1,D2... ,Dn), then D1*D2... *DN<=1000000,n<=8

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator FloorMod.
*/
REG_OP(FloorMod)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(FloorMod)

} // namespace ge

#endif // OPS_OP_PROTO_INC_FLOOR_MOD_H_
