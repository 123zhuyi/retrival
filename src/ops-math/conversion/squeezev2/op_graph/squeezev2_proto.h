/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file squeezev2_proto.h
 * \brief
 */
#ifndef SQUEEZEV2_PROTO_H_
#define SQUEEZEV2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Removes dimensions of size 1 from the shape of a tensor. \n

*@par Inputs:
*x: A tensor. All data types are supported. \n

*@par Attributes:
*axis: An optional list of int32 or int64. Defaults to []. If not specified, squeezes all dimensions of size 1.
If specified, only squeezes the dimensions listed. It is an error to squeeze a dimension that is not 1. \n

*@par Outputs:
*y: A tensor. The same type as input x. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Squeeze.

*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Squeeze instead.
*/
REG_OP(SqueezeV2)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(SqueezeV2)
} // namespace ge
#endif // SQUEEZEV2_PROTO_H_