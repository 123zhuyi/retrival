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
 * \file depth_to_space_proto.h
 * \brief
 */
#ifndef OP_PROTO_DEPTH_TO_SPACE_PROTO_H_
#define OP_PROTO_DEPTH_TO_SPACE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Rearranges data from depth into blocks of spatial data .

* @par Inputs:
* x: A Tensor. The data type must be one of BasicType.
* The data format must be NCHW or NHWC and must be same as the attribute value data_format.

* @par Attributes:
* Three attributes, including:
* @li block_size: An int >= 2, specifying the size of the spatial block.
* @li mode: An optional string, specifying the mode. Must be DCR(depth-column-row)
*     or CRD(column-row-depth). Defaults to "DCR".
* @li data_format: An optional string, specifying the data format. Must be
*     NCHW or NHWC, and be same as the data format of x. Defaults to "NHWC".

* @par Outputs:
* y: A Tensor of the same type as "x".

* @par Third-party framework compatibility:
* Compatible with TensorFlow operator DepthToSpace.
*/
REG_OP(DepthToSpace)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .ATTR(mode, String, "DCR")
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(DepthToSpace)

} // namespace ge

#endif // OP_PROTO_DEPTH_TO_SPACE_PROTO_H_
