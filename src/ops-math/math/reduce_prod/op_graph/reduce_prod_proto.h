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
 * \file reduce_prod_proto.h
 * \brief
 */

#ifndef REDUCE_PROD_PROTO_H_
#define REDUCE_PROD_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief  Reduce a tensor on a certain axis based on product.

*@par Inputs:
*Two inputs, including:
*@li x: A Tensor. Must be the type of NumberType.(NumberType
*includes: complex128, complex64, double, float32, float16, int16,
*int32, int64,int8, qint32, qint8, quint8, uint16, uint32, uint64,
*uint8, bfloat16, complex32).Supported format list ["ND"].
*@li axes: A Tensor. Must be the type of IndexNumberType(
*includes: int32, int64). The dimensions to reduce.Supported format list ["ND"]. \n

*@par Attributes:
*keep_dims: A bool. If true, retains reduced dimensions with length 1.
*Optional and defaults to "False" . \n
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceProd.
*/
REG_OP(ReduceProd)
    .INPUT(x,TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y,TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceProd)

} // namespace ge

#endif