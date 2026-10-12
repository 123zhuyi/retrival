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
 * \file as_strided_proto.h
 * \brief
 */
#ifndef OP_PROTO_AS_STRIDED_H_
#define OP_PROTO_AS_STRIDED_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief  Make memory of a view be contiguous.

*@par Inputs:
*Four inputs, including:
*@li x: The input tensor. Must be the type of hifloat8, float8_e5m2,
*float8_e4m3fn, bool and BasicType. Support format "ND".
*@li size: The shape of output tensor. Must be the type of
*IndexNumberType(IndexNumberType includes: int32, int64). 
*All elements in the size must be non-negative integers. 
*Support format "ND".
*@li stride: The stride of output tensor. Must be the type of IndexNumberType.
*All elements in the stride must be non-negative integers. 
*Support format "ND".
*@li storage_offset: The offset in the underlying storage of the output tensor.
*Must be a non-negative interger.
*Must be the type of IndexNumberType. Support format "ND".

*@par Outputs:
*y: A Tensor. Has the same type as "x". Support format "ND".

*@par Third-party framework compatibility
*Compatible with the PyTorch operator as_strided.
*/
REG_OP(AsStrided)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_BOOL}))
    .INPUT(size, TensorType::IndexNumberType())
    .INPUT(stride, TensorType::IndexNumberType())
    .INPUT(storage_offset, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_BOOL}))
    .OP_END_FACTORY_REG(AsStrided)

} // namespace ge
#endif // OP_PROTO_AS_STRIDED_H_