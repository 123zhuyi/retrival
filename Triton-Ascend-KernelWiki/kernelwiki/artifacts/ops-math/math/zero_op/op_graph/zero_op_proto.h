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
 * \file zeros_like_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ZEROS_LIKE_H_
#define OPS_OP_PROTO_INC_ZEROS_LIKE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns a tensor of the same dtype and shape as the input tensor with all elements set to zero.

*@par Inputs:
*x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types: BasicType() and variant.
*
*@par Outputs:
*y: A ND Tensor of the same data type as "x".
*
*@attention Constraints:
* The output has the same shape and type as the input.
* The dtype DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1 only support version since Ascend950
*
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator zeros_like.
*/
REG_OP(ZerosLike)
    .INPUT(x, TensorType({BasicType(), DT_VARIANT, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({BasicType(), DT_VARIANT, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OP_END_FACTORY_REG(ZerosLike)
} // namespace ge

#endif // OPS_OP_PROTO_INC_ZEROS_LIKE_H_
