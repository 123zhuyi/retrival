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
 * \file identity_proto.h
 * \brief Definition of the Identity operator.
 */
#ifndef IDENTITY_PROTO_H_
#define IDENTITY_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge
{
/**
*@brief Return a tensor with the same shape and contents as input. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

*@par Outputs:
*y: A tensor with the same shape、data type and contents as input. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Identity.
*/
REG_OP(Identity) 
    .INPUT(x, TensorType::ALL()) 
    .OUTPUT(y, TensorType::ALL()) 
    .OP_END_FACTORY_REG(Identity)
} // namespace ge
#endif // IDENTITY_PROTO_H_