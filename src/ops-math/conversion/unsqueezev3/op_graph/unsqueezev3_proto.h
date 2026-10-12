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
 * \file unsqueezev3_proto.h
 * \brief
 */
#ifndef UNSQUEEZEV3_PROTO_H_
#define UNSQUEEZEV3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape
is changed, but the data is not changed. \n

*@par Inputs:
*x: A tensor. All data types are supported. \n
*axes: A list of int64, which indicates the dimensions to be inserted. \n

*@par Outputs:
*y: Reshape tensor with same data as input. The same type as input x. \n

*@par Third-party framework compatibility
*Compatible with the Onnx operator Unsqueeze in V3. \n
*/

REG_OP(UnsqueezeV3)
    .INPUT(x, TensorType::ALL())
    .INPUT(axes, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(UnsqueezeV3)

/**
*@brief Creates a constant tensor for training. \n

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type. \n

*@par Outputs:
*y: The constant tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Const.
*/
REG_OP(Constant)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT4, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Constant)
} // namespace ge
#endif // UNSQUEEZEV3_PROTO_H_