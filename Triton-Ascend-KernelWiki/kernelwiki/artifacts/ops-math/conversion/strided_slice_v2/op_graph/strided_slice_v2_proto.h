/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OP_PROTO_STRIDED_SLICE_V2_H_
#define OP_PROTO_STRIDED_SLICE_V2_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
* @brief Extracts a strided slice of a tensor. Roughly speaking, this op
*   extracts a slice of size (end-begin)/stride from the given input tensor.
*   Starting at the location specified by begin the slice continues by
*   adding stride to the index until all dimensions are not less than end. \n
*
* @par Inputs:
* Five inputs, including:
* @li x: A Tensor. 1-8 dimensions. Must be one of the following types:
* double, float32, float16, bfloat16, complex32, complex64, complex128,
* int8, uint8, int16, uint16, int32, uint32, int64, uint64, qint8, quint8, qint16, quint16, qint32, bool, hifloat8,
* float8_e5m2, float8_e4m3fn, float8_e8m0.
* @li begin: A Tensor of type int32 or int64, for the index of the first value to select.
*     Elements in begin with negative values are interpreted as indices from the end of the dimension.
* @li end: A Tensor of type int32 or int64, for the index of the last value to select.
*     Elements in end with negative values are interpreted as indices from the end of the dimension.
* @li axes: A Tensor of type int32 or int64, indicate axis to be select.
*     When not provided, slices all dimensions.
* @li strides: A Tensor of type int32 or int64, for the increment.
*     When not provided, stride defaults to 1. All elements in strides must be non-zero integers. \n
*
* @par Attributes:
* @li begin_mask: An attribute of type Int.
*     A bitmask where a bit "i" being "1" means to ignore the begin
*     value and instead use the largest interval possible. Default value is 0.
* @li end_mask: An attribute of type Int.
*     Analogous to "begin_mask". Default value is 0.
* @li ellipsis_mask: An attribute of type Int.
*     A bitmask where bit "i" being "1" means the "i"th position
*     is actually an ellipsis. Default value is 0.
* @li new_axis_mask: An attribute of type Int.
*     A bitmask where bit "i" being "1" means the "i"th
*     specification creates a new shape 1 dimension. Default value is 0.
* @li shrink_axis_mask: An attribute of type Int.
*     A bitmask where bit "i" implies that the "i"th
*     specification should shrink the dimensionality. Default value is 0. \n
*
* @par Outputs:
* y: A Tensor that has the same type as "x".
*
* @attention Constraints:
*
* @par Third-party framework compatibility
* Compatible with the onnx operator Slice.
 */
REG_OP(StridedSliceV2)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .OP_END_FACTORY_REG(StridedSliceV2)
} // namespace ge
#endif // OP_PROTO_STRIDED_SLICE_V2_H_
