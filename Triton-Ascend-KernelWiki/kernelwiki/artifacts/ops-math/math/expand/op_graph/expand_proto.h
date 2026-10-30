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
 * \file expand_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_EXPAND_H_
#define OPS_OP_PROTO_INC_EXPAND_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Expand the input tensor to a compatible shape. \n

* @par Inputs:
* One inputs, including:
* @li x: A Tensor. Must be one of the following types:
*     float16, float32, int32, int64, int8, uint8, bool, bfloat16. \n
* @li shape: A Tensor to specify the shape that the input tensor expanded to. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x", and the shape specified by input and attr shape \n

* @par Third-party framework compatibility
* Compatible with the ONNX operator Expand.
*
* @attention Constraints:
* @li The dim numbers of shape cannot be more than one.
* @li The inputs cannot be empty tensor.
*/

REG_OP(Expand)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32,DT_INT64, DT_INT8, DT_UINT8, DT_BOOL, DT_BF16}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32,DT_INT64, DT_INT8, DT_UINT8, DT_BOOL, DT_BF16}))
    .OP_END_FACTORY_REG(Expand)

} // namespace ge

#endif // OPS_OP_PROTO_INC_EXPAND_H_

