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
 * \file neg_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_NEG_H_
#define OPS_OP_PROTO_INC_NEG_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes numerical negative value element-wise (y = -x)

*@par Inputs:
* One input:
*x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* float16, float32, int32, int64, complex64, complex128, bfloat16, int8, float64.

*@par Outputs:
*y: A ND Tensor. Has the same dtype and format as input "x".

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Neg.
*/
REG_OP(Neg)
    .INPUT(
        x, TensorType(
               {DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(
        y, TensorType(
               {DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Neg)

} // namespace ge

#endif // OPS_OP_PROTO_INC_NEG_H_
