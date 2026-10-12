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
 * \file adds_proto.h
 * \brief Adds 算子 GE IR 原型注册（图模式）
 *
 * 输入 x，输出 y，属性 value (Float, 必选)。
 */
#ifndef OPS_OP_PROTO_INC_ADDS_H_
#define OPS_OP_PROTO_INC_ADDS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief Add tensor with value.

 *@par Inputs:
 *One input, including: \n
 * x: A ND Tensor. Must be one of the following types:int32,int16, float16, float32, bfloat16, int64. \n

 *@par Attributes:
 *value: A scale. Must be float. \n

 *@par Outputs:
 *y: A ND Tensor. Has the same dtype and shape as "x1". \n

 *@par Third-party framework compatibility:
 * Compatible with the PyTorch operator adds.
 *@attention Constraints:
 * For parameters of the float32 type, there is no precision loss. For INT32 and INT64 parameters,
 * precision loss occurs when the parameter value exceeds 2^24. it is recommended to use Add.
 */
REG_OP(Adds)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_INT64}))
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(Adds)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ADDS_H_
