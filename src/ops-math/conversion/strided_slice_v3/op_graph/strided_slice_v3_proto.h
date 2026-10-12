/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_STRIDED_SLICE_V3_H_
#define OP_PROTO_STRIDED_SLICE_V3_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
* @brief Extracts a strided slice of a tensor. Roughly speaking, this op
    extracts a slice of size (end-begin)/stride from the given input tensor.
    Starting at the location specified by begin the slice continues by
    adding stride to the index until all dimensions are not less than end.

* @par Inputs:
* Five inputs, including:
* @li x: A Tensor. 1-8 dimensions. Must be one of the following types: float32, float64, int32, uint8, int16, int8,
*     complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, float16, bfloat16, uint32, uint64, bool, hifloat8, float8_e5m2, float8_e4m3fn, float8_e8m0.
* @li begin: A Tensor of type int32 or int64, for the index of the first value to select.
*     Elements in begin with negative values are interpreted as indices from the end of the dimension. 
*     Negative indices are clamped to the valid range after conversion.
*     If the length of begin is less than the number of axes, the missing values default to 0. \n
* @li end: A Tensor of type int32 or int64, for the index of the last value to select.
*     Elements in end with negative values are interpreted as indices from the end of the dimension.
*     Negative indices are clamped to the valid range after conversion.
*     If the length of end is less than the number of axes, the missing values default to the input dimension size. \n
* @li strides: Optional. A Tensor of type int32 or int64, for the increment.
*     When not provided, stride defaults to 1.
*     If the length of strides is less than the number of axes, the missing values default to 1. \n
* @li axes: Optional. A Tensor of type int32 or int64, specifying the axes to apply slicing.
*     Each element in axes must be in the range [-rank(x), rank(x)-1].
*     When not provided, slices all dimensions [0, 1, 2, ..., rank(x)-1]. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @attention Constraints:
* @li At least one valid axis value must be provided (either via axes tensor or default).

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(StridedSliceV3)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .OP_END_FACTORY_REG(StridedSliceV3)
} // namespace ge
#endif // OP_PROTO_STRIDED_SLICE_V3_H_
