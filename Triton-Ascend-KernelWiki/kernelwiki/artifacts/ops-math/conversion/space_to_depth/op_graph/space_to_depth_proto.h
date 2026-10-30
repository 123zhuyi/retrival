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
 * \file transpose_proto.h
 * \brief
 */
#ifndef SPACE_TO_DEPTH_PROTO_H
#define SPACE_TO_DEPTH_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Outputs a copy of the input tensor where values from the "height" and
* "width" dimensions are moved to the "depth" dimension . \n

* @par Inputs:
* x: A Tensor. The data type must be one of BasicType.
* The data format must be NCHW or NHWC and must be same as the attribute value data_format.

* @par Attributes:
* @li block_size: A required int, specifying the input block size.
* @li data_format: An optional string, specifying the data format. Must be
*     NCHW or NHWC, and be same as the data format of x. Defaults to "NHWC".

* @par Outputs:
* y: A Tensor. Has the same type as input "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToDepth.
*/
REG_OP(SpaceToDepth)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(SpaceToDepth)
} // namespace ge
#endif // SPACE_TO_DEPTH_PROTO_H
