/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reshape_proto.h
 * \brief
 */
#ifndef OPS_OP_RESHAPE_PROTO_H_
#define OPS_OP_RESHAPE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Reshapes a tensor without changing its underlying data. \n

*@par Inputs:
*@li x: A tensor. \n
*@li shape: A tensor that defines the output shape. Must be int32 or int64. \n

*@par Attributes:
*@li axis: An optional int. The first dimension to reshape. Defaults to 0. \n
*@li num_axes: An optional int. The extent of the reshape. Defaults to -1. \n

*@par Outputs:
*@li y: A tensor with the same type as x. \n

*@attention Constraints:
*Reshape runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator Reshape.
*@li Compatible with the Caffe operator Reshape.
*@li Compatible with the ONNX operator Reshape.
*/
REG_OP(Reshape)
    .INPUT(x, TensorType::ALL())
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)
} // namespace ge

#endif // OPS_OP_RESHAPE_PROTO_H_