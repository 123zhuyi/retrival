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
 * \file gcd_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_GCD_H_
#define OPS_OP_PROTO_INC_GCD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns x1 and x2 greatest common divisor element-wise. 
* Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor, support 1D ~ 8D.
* Must be one of the following types: uint8, int8, int16, int32, int64.
* @li x2: A ND Tensor of the same dtype as "x1", support 1D ~ 8D. \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype as "x1". \n

* @par Third-party framework compatibility
* Compatible with the Torch operator Gcd.
*/
REG_OP(Gcd)
    .INPUT(x1, "T")
    .INPUT(x2, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Gcd)

} // namespace ge

#endif // OPS_OP_PROTO_INC_GCD_H_
