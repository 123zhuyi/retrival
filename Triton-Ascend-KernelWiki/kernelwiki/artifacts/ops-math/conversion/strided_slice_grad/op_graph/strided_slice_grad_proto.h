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
 * \file selection_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Since StridedSlice cuts out pieces of its "input" which is size "dy",
    its gradient will have the same shape (which is passed here as "shape").
    The gradient will be zero in any element that the slice does not select .

* @par Inputs:
* Five inputs, including:
* @li shape: A Tensor of type int32 or int64.
* @li begin: A Tensor of type int32 or int64.
      The index of the first value to select.
* @li end: A Tensor of type int32 or int64.
      The index of the last value to select.
* @li strides: A Tensor of type int32 or int64, for the increment.
* @li dy: A Tensor. Supported dtype is BasicType. \n

* @par Attributes:
* @li begin_mask: A Tensor of type int32.
      A bitmask where a bit "i" being "1" means to ignore the begin
      value and instead use the largest interval possible.
* @li end_mask: A Tensor of type int32.
      Analogous to "begin_mask".
* @li ellipsis_mask: A Tensor of type int32.
      A bitmask where bit "i" being "1" means the "i"th position
      is actually an ellipsis.
* @li new_axis_mask: A Tensor of type int32.
      A bitmask where bit "i" being "1" means the "i"th
      specification creates a new shape 1 dimension.
* @li shrink_axis_mask: A Tensor of type int32.
      A bitmask where bit "i" implies that the "i"th
      specification should shrink the dimensionality . \n

* @par Outputs:
* output: A Tensor has the same type as "dy" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator StridedSliceGrad.
*/
REG_OP(StridedSliceGrad)
    .INPUT(shape, TensorType::IndexNumberType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .INPUT(dy, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceGrad)
}

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_