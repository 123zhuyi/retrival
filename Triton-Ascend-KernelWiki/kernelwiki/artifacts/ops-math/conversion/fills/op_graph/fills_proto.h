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
 * \file fills_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Fill tensor with value.

*@par Inputs:
*One input, including:
* x: A ND tensor. Must be one of the following types:BasicType, bool.

*@par Outputs:
*y: A ND tensor. Has the same dtype and shape as "x". \n

*@par Attributes:
*value: A scale. Must be float. \n

*@par Third-party framework compatibility:
* Compatible with the PyTorch operator fills.
*@attention Constraints:
* For parameters of the float32 type, there is no precision loss. For INT32 and INT64 parameters,
* precision loss occurs when the parameter value exceeds 2^24. it is recommended to use Fill.
*/
REG_OP(Fills)
     .INPUT(x, TensorType({BasicType(), DT_BOOL}))
     .OUTPUT(y, TensorType({BasicType(), DT_BOOL}))
     .REQUIRED_ATTR(value, Float)
     .OP_END_FACTORY_REG(Fills)
}

#endif