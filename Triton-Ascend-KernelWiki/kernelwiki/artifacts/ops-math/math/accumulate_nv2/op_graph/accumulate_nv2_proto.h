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
 * \file accumulate_nv2_proto.h
 * \brief
 */
#ifndef ACCUMULATE_NV2_PROTO_H_
#define ACCUMULATE_NV2_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Returns the element-wise sum of a list of tensors.\n
* AccumulateNV2 performs the same operation as AddN, but does not wait for all of its inputs
to be ready before beginning to sum.\n This can save memory if inputs are ready at different times,
since minimum temporary storage is proportional to the output size rather than the inputs size.
 Returns a Tensor of same shape and type as the elements of inputs.

*
*@par Inputs:
*Dynamic inputs, including:
* x: A tensor. Must be one of the following types: float16, float32, int32, int8, uint8. It's a dynamic input. \n
*
*@par Outputs:
* y: A tensor. Has the same dtype as "x".
*
*@par Attributes:
* N: the size of x. Must be "int".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AccumulateNV2.
*
*/
REG_OP(AccumulateNV2)
   .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
   .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
   .REQUIRED_ATTR(N, Int)
   .OP_END_FACTORY_REG(AccumulateNV2)

}  // namespace ge

#endif  // ARG_MAX_V2_PROTO_H_
