/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reciprocal_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_RECIPROCAL_H_
#define OPS_OP_PROTO_INC_RECIPROCAL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes the reciprocal of "x".

*@par Inputs:
*One inputs, include:
*x:A ND Tensor of type float16, float32, double,
*     complex64, complex128, bfloat16. the format can be [NCHW,NHWC,ND]

*@par Outputs:
*y:A ND Tensor with same type as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Reciprocal.
*/
REG_OP(Reciprocal)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16,
                          DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .OP_END_FACTORY_REG(Reciprocal)

} // namespace ge

#endif // OPS_OP_PROTO_INC_RECIPROCAL_H_

