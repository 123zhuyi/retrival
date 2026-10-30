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
 * \file strided_slice_proto.h
 * \brief
 */
#ifndef OP_PROTO_STRIDED_PROTO_H_
#define OP_PROTO_STRIDED_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
* @brief Extracts a strided slice of a tensor. Roughly speaking, this op
    extracts a slice of size (end-begin)/stride from the given input tensor.
    Starting at the location specified by begin the slice continues by
    adding stride to the index until all dimensions are not less than end.

* @par Inputs:
* Four inputs, including:
* @li x: A tensor. Must be one of the BasicType: complex128, complex64,
  double, float32, float16, int16, int32, int64, int8, qint16, qint32, qint8,
  quint16, quint8, uint16, uint32, uint64, uint8, bfloat16, complex32,
  hifloat8, float8_e5m2, float8_e4m3fn.Supported format list ["ND"].

* @li begin: A tensor of IndexNumberType: int32 or int64,
  for the index of the first value to select.Supported format list ["ND"].

* @li end: A tensor of IndexNumberType: int32 or int64,
  for the index of the last value to select.Supported format list ["ND"].

* @li strides: A tensor of IndexNumberType: int32 or int64,
  for the increment.Supported format list ["ND"].

* @par Attributes:
* @li begin_mask: A tensor of type int includes all types of int.
      A bitmask where a bit "i" being "1" means to ignore the begin
      value and instead use the largest interval possible.Default value is 0.
* @li end_mask: A tensor of type int includes all types of int.
      Analogous to "begin_mask".Default value is 0.
* @li ellipsis_mask: A tensor of type int includes all types of int.
      A bitmask where bit "i" being "1" means the "i"th position
      is actually an ellipsis.Default value is 0.
* @li new_axis_mask: A tensor of type int includes all types of int.
      A bitmask where bit "i" being "1" means the "i"th
      specification creates a new shape 1 dimension.Default value is 0.
* @li shrink_axis_mask: A tensor of type int includes all types of int.
      A bitmask where bit "i" implies that the "i"th
      specification should shrink the dimensionality.Default value is 0.

* @par Outputs:
* y: A tensor. Has the same type as "x".Supported format list ["ND"].

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator StridedSlice.
*/
REG_OP(StridedSlice)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(StridedSlice)

} // namespace ge
#endif // OP_PROTO_STRIDED_PROTO_H_