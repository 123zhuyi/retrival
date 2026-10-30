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
 * \file circular_pad_proto.h
 * \brief
 */
#ifndef OP_PROTO_CIRCULAR_PAD_PROTO_H_
#define OP_PROTO_CIRCULAR_PAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
namespace ge {
/**
* @brief Pads the input tensor using circular padding of the input boundary.
* Tensor values at the beginning of the dimension are used to pad the end,
* and values at the end are used to pad the beginning.
* If negative padding is applied then the ends of the tensor get removed.
*
* @par Inputs:
* @li x: A tensor, input tensor.
* @li paddings: A tensor, the size of the padding.
*
* @par Outputs:
* y: A tensor, output tensor.
*
* @par Third-party framework compatibility
* Compatible with the PyTorch operator CircularPad2d and CircularPad3d.
*
* @platform: KirinX90, Kirin9030
*
*/
REG_OP(CircularPad)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32}))
    .INPUT(paddings, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(CircularPad)

}  // namespace ge

#endif  // OP_PROTO_CIRCULAR_PAD_PROTO_H_