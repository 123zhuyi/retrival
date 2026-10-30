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
 * \file mul_no_nan_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MUL_NO_NAN_H_
#define OPS_OP_PROTO_INC_MUL_NO_NAN_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Element-wise multiply that returns 0 when x2 == 0, masking the
*        IEEE-754 0 * inf = NaN and 0 * NaN = NaN traps. Supports NumPy
*        broadcasting between x1 and x2.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND tensor. Must be one of the following types: float16, float32, int32, bfloat16.
* @li x2: A ND tensor. Must have the same dtype as x1; shape must be broadcastable with x1. \n

* @par Outputs:
* y: A ND tensor. Has the same dtype as x1; shape is the broadcast shape of x1 and x2.
*    y[i] = 0          if x2[i] == 0
*    y[i] = x1[i]*x2[i] otherwise \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MulNoNan (tf.math.multiply_no_nans).
*/
REG_OP(MulNoNan)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(MulNoNan)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MUL_NO_NAN_H_
