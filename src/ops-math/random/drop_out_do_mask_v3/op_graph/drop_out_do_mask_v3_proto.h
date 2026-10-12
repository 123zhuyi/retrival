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
 * \file drop_out_do_mask_v3_proto.h
 * \brief
 */
#ifndef DROP_OUT_DO_MASK_V3_PROTO_H_
#define DROP_OUT_DO_MASK_V3_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Return "output" according to the algorithm of dropout_do_mask_v3:  \n
*  scale_x = x *(1 / keep_prob)  \n
*  output = select(mask == 1, scale_x, 0)  \n

* @par Inputs:
* Three inputs, including:
* @li x: A mutable Tensor. A ND tensor. Support 1D ~ 8D. Must be one of the following types:
*     float16, float32, bfloat16.
* @li mask: A mutable Tensor. A ND tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8 or bool.
*     value of shape should met the following algorithm:
*     value = value = ((size(x) + 127) / 128) * 128 / 8.
* @li keep_prob: A mutable Tensor.Must met all of the following rules:
*     shape of "keep_prob" should be (1,).
*     Has the same type as "x" . \n

* @par Outputs:
* y: A mutable Tensor. Has the same type, shape and format as "x".
*/
REG_OP(DropOutDoMaskV3)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_BOOL}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(DropOutDoMaskV3)

} // namespace ge

#endif // DROP_OUT_DO_MASK_V3_PROTO_H_
