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
 * \file muls_proto.h
 * \brief
 */

#ifndef MULS_PROTO_H_
#define MULS_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Multiply tensor with scale.

*@par Inputs:
*One input, including:
* x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types:
* bfloat16, int32, int16, int64, float16, float32, complex32, complex64.
*
*@par Outputs:
* y: A ND Tensor. Has the same dtype and shape as "x".
*
*@par Attributes:
*@li value: An required attribute. Must be float.
*
*@par Third-party framework compatibility:
* Compatible with the PyTorch operator muls.
*@attention Constraints:
* For parameters of the float32 type, there is no precision loss. For INT32 and INT64 parameters,
* precision loss occurs when the parameter value exceeds 2^24. it is recommended to use Mul.
*/
REG_OP(Muls)
     .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
     .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
     .REQUIRED_ATTR(value, Float)
     .OP_END_FACTORY_REG(Muls)

}  // namespace ge

#endif  // MULS_PROTO_H_
