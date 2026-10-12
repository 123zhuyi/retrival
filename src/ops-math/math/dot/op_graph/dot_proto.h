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
 * \file dot_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_DOT_H_
#define OPS_OP_PROTO_INC_DOT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the dot product (inner product) of two tensors. This function does not broadcast.

* @par Inputs:
* Two inputs, including:
* @li input_x: A ND Tensor. the first tensor must be 1d. Must be one of the following types:
* float32, float16, bfloat16, uint8, int8, int32. \n
* @li input_y: A ND Tensor. the second tensor must be 1d. Must be one of the following types:
* float32, float16, bfloat16, uint8, int8, int32. \n

* @par Outputs:
* output: A ND Tensor. Result of the two inputs, must be 1d. A ND Tensor of the same dtype as input_x. \n

* @par Third-party framework compatibility
* Compatible with the PyTorch dot operator. \n
*/
REG_OP(Dot)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT32}))
    .INPUT(input_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT32}))
    .OP_END_FACTORY_REG(Dot)

} // namespace ge

#endif // OPS_OP_PROTO_INC_DOT_H_

