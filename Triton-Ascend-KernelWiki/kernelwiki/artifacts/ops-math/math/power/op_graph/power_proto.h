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
 * \file power_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_POWER_H_
#define OPS_OP_PROTO_INC_POWER_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes y = exp(power * log(x * scale + shift)) element-wise.
*
* @par Inputs:
* One input:
* x: An ND tensor. Must be one of the following types: bfloat16, float16, float32.
*
* @par Attributes:
* @li power: A required attribute of type float32, specifying the exponent.
* @li scale: An optional attribute of type float32, specifying the scale factor. Defaults to "1.0".
* @li shift: An optional attribute of type float32, specifying the shift bias. Defaults to "0.0".
*
* @par Outputs:
* y: A ND tensor of the same dtype as "x".
*/
REG_OP(Power)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(power, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Power)

} // namespace ge

#endif // OPS_OP_PROTO_INC_POWER_H_
