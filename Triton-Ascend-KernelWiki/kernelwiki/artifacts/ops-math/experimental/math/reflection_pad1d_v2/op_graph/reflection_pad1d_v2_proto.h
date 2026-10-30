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
 * \file reflection_pad1d_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_REFLECTION_PAD1D_V2_OPS_H_
#define OPS_OP_PROTO_INC_REFLECTION_PAD1D_V2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Performs reflection padding on the input tensor according to the specified padding sizes.
 *
 * Mirror padding (also known as reflection padding) fills the edges of the input tensor by reflecting
 * the values from the tensor's boundary along each dimension, as specified by the `paddings` tensor.
 * Unlike constant padding, the filled values are derived from the input tensor itself in a mirrored manner.
 *
 *@par Inputs:
 * Two inputs, including:
 *@li x: A Tensor. The input tensor to be padded.
 * Type must be one of the following types: float16 (DT_FLOAT16), float32 (DT_FLOAT), bfloat16 (DT_BF16).
 *@li paddings: A Tensor. Specifies the padding size for each dimension of the input tensor.
 * The shape of `paddings` must be [num_dims, 2], where `num_dims` is the rank of input `x`.
 * Each row [pad_left, pad_right] indicates the number of elements to pad on the left (before) and right (after)
 * the corresponding dimension of `x`. Type must be one of the following types: float16 (DT_FLOAT16), float32
 * (DT_FLOAT), bfloat16 (DT_BF16). \n
 *
 *@par Outputs:
 *y: A Tensor. The padded output tensor with the same data type as input `x`.
 * The shape of `y` is determined by the input shape plus the padding sizes:
 * for each dimension i, output_shape[i] = input_shape[i] + paddings[i][0] + paddings[i][1]. \n
 *
 *@par Third-party framework compatibility
 * Compatible with PyTorch operator `torch.nn.functional.pad` (mode='reflect') / `torch.nn.ReflectionPad*`,
 * and TensorFlow operator `tf.pad` (mode='REFLECT') / `tf.raw_ops.ReflectionPad1dV2`.
 */
REG_OP(ReflectionPad1dV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(paddings, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(ReflectionPad1dV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_REFLECTION_PAD1D_V2_OPS_H_