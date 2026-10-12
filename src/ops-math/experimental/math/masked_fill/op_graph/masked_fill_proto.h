/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file masked_fill_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MASKED_FILL_OPS_H_
#define OPS_OP_PROTO_INC_MASKED_FILL_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Replaces elements in the input tensor where the corresponding mask value is True with a specified value.

*@par Inputs:
* Three inputs, including:
*@li x: A Tensor. The input tensor to be processed.
* Type must be one of the following types:
* float16, float32, float64, uint8, uint16, uint32, uint64, int8, int16, int32, int64, bfloat16.
*@li mask: A Tensor of type bool. The mask tensor where True indicates the position in `x` to be replaced.
* The shape of `mask` must be broadcastable to the shape of `x`.
*@li value: A Scalar. The value to replace elements in `x` where `mask` is True.
* Type must be the same as `x`. \n

*@par Outputs:
*z: A Tensor. The output tensor with the same shape and type as `x`, where elements at positions specified by `mask`
(True) are replaced with `value`. \n

*@par Third-party framework compatibility
* Compatible with the PyTorch operator masked_fill and TensorFlow operator where.
*/
REG_OP(MaskedFill)
    .INPUT(
        x, TensorType(
               {DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_INT32,
                DT_INT64, DT_BF16}))
    .INPUT(mask, TensorType({DT_BOOL}))
    .INPUT(
        value, TensorType(
                   {DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16,
                    DT_INT32, DT_INT64, DT_BF16}))
    .OUTPUT(
        z, TensorType(
               {DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_INT32,
                DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(MaskedFill)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MASKED_FILL_OPS_H_