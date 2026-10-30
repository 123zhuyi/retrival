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
 * \file drop_out_do_mask_v3_d_proto.h
 * \brief
 */
#ifndef DROP_OUT_DO_MASK_V3_D_PROTO_H_
#define DROP_OUT_DO_MASK_V3_D_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Return "output" according to the algorithm of dropout_do_mask_v3_d:  \n
*  scale_x = x *(1 / keep_prob)  \n
*  output = select(mask == 1, scale_x, 0)  \n

* @par Inputs:
* Two inputs, including:
* @li x: A mutable Tensor. A ND tensor. Support 1D ~ 8D. Must be one of the following types:
*     float16, float32, bfloat16.
* @li mask: A mutable Tensor. A ND tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8 or bool.
*     value of shape should met the following algorithm:
*     value = value = ((size(x) + 127) / 128) * 128 / 8.

* @par Attributes:
* @li keep_prob: A required Float attribute. Must satisfy 0 <= keep_prob <= 1.

* @par Outputs:
* y: A mutable Tensor. Has the same type, shape and format as "x".
*/
REG_OP(DropOutDoMaskV3D)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(keep_prob, Float)
    .OP_END_FACTORY_REG(DropOutDoMaskV3D)

} // namespace ge

#endif // DROP_OUT_DO_MASK_V3_D_PROTO_H_
