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
 * \file pow_proto.h
 * \brief
 */

#ifndef OP_PROTO_POW_PROTO_H_
#define OP_PROTO_POW_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
namespace ge {
/**
* @brief Computes the pow of a tensor.

*@par Inputs:
* @li x: A tensor of type float16, float32, bfloat16, int8, uint8, int16, int32.
* @li exponent: A tensor of type float16, float32, bfloat16, int8, uint8, int16, int32.
*@par Outputs:
* y: A tensor of type float16, float32, bfloat16, int8, uint8, int16, int32.

*@par Third-party framework compatibility
* Compatible with the Pytorch operator Pow.

*@par Restrictions:
* @li x and exponent must have the same dtype.
* @li x and exponent must be broadcast-compatible (following PyTorch's broadcasting rules).
* @li The output tensor y must have the same shape as the broadcast result of x and exponent.
* @li If x is integer type and exponent is negative, the behavior is undefined (consistent with PyTorch).
*/
#define POW_TYPES DT_BF16, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32

REG_OP(Pow)
    .INPUT(x, TensorType({POW_TYPES}))
    .INPUT(exponent, TensorType({POW_TYPES}))
    .OUTPUT(y, TensorType({POW_TYPES}))
    .OP_END_FACTORY_REG(Pow);

} // namespace ge

#endif // OP_PROTO_POW_PROTO_H_