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
 * \file slice_with_axes_proto.h
 * \brief
 */
#ifndef OPS_OP_SLICE_WITH_AXES_PROTO_H_
#define OPS_OP_SLICE_WITH_AXES_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Extracts a slice from a tensor along specified axes. \n

*@par Inputs:
*@li x: A tensor of BasicType.
*@li offsets: A 1D tensor of type int32 or int64. The start offsets for each axis.
*@li size: A 1D tensor of type int32 or int64. The sizes for each axis. \n

*@par Outputs:
*y: A tensor with the same type as x. \n

*@par Attributes:
*axes: Required. List of axes along which to slice. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Slice.
*/
REG_OP(SliceWithAxes)
    .INPUT(x, TensorType::BasicType())
    .INPUT(offsets, TensorType::IndexNumberType())
    .INPUT(size, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(axes, ListInt)
    .OP_END_FACTORY_REG(SliceWithAxes)
} // namespace ge
#endif
