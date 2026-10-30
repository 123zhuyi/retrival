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
 * \file bitwise_and_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_BITWISE_AND_H_
#define OPS_OP_PROTO_INC_BITWISE_AND_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Element-wise computes the bitwise AND of "x1" and "x2". Support broadcasting operations.

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: int8, int16,
*     int32, int64, uint8, uint16, uint32, uint64. Broadcasting is supported.
* @li x2: A ND Tensor of the same dtype as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BitwiseAnd.
*/
REG_OP(BitwiseAnd)
    .INPUT(x1, TensorType::IntegerDataType())
    .INPUT(x2, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(BitwiseAnd)

} // namespace ge

#endif // OPS_OP_PROTO_INC_BITWISE_AND_H_

