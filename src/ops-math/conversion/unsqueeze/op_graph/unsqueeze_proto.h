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
 * \file unsqueeze_proto.h
 * \brief
 */
#ifndef UNSQUEEZE_PROTO_H_
#define UNSQUEEZE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*x: Original tensor. All data types are supported. \n

*@par Attributes:
*axes: List of ints indicating the dimensions to be inserted. Defaults to []. \n

*@par Outputs:
*y: Reshape tensor with same data as input. The same type as input x. \n

*@par Third-party framework compatibility
*Compatible with the Onnx operator Unsqueeze.
*/

REG_OP(Unsqueeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axes, ListInt, {})
    .OP_END_FACTORY_REG(Unsqueeze)
} // namespace ge
#endif // UNSQUEEZE_PROTO_H_