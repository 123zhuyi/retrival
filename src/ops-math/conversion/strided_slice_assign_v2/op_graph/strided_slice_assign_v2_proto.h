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
 * \file strided_slice_assign_v2_proto.h
 * \brief
 */

#ifndef STRIDED_SLICE_ASSIGN_V2_PROTO_H
#define STRIDED_SLICE_ASSIGN_V2_PROTO_H
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Assigns "value" to the sliced l-value reference of "var".
* The values of "value" are assigned to the positions in the variable. "var"
* that are selected by the slice parameters. The slice parameters "begin, "end",
* "strides", etc. work exactly as in "StridedSlice" . \n

* @par Inputs:
* Five inputs, including:
* @li var: A mutable ND Tensor of type BasicType.
* Support dtype: [float16, float32, bfloat16, int32, int64, double, int8], Supoort format: [ND]. \n
* @li input_value: A mutable ND Tensor of type BasicType .
* Support dtype: [float16, float32, bfloat16, int32, int64, double, int8], Supoort format: [ND]. \n
* @li begin: A mutable ND Tensor of type IndexNumberType.
* Support dtype: [int64], Supoort format: [ND]. \n
* Specifies the index of the first value to select.
* @li end: A mutable ND Tensor of type IndexNumberType.
* Support dtype: [int64], Supoort format: [ND]. \n
* Specifies the index of the last value to select.
* @li strides: A mutable ND Tensor of type IndexNumberType.
* Support dtype: [int64], Supoort format: [ND]. \n
* Specifies the stride to select.
* @li axes: Optional. A mutable ND Tensor of type IndexNumberType.
* Support dtype: [int64], Supoort format: [ND]. \n
* Specifies the stride to select. \n

* @par Outputs:
* var: A mutable Tensor. Has the same type and format as "var" . \n

* @attention Constraints:
* This operator currently does not support broadcasting. Therefore, the shape
* of "value" must be exactly the shape produced by the slice of "var" . \n

* @see StridedSlice()

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator StridedSlice.
*/
REG_OP(StridedSliceAssignV2)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_DOUBLE, DT_INT8}))
    .INPUT(input_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_DOUBLE, DT_INT8}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_DOUBLE, DT_INT8}))
    .OP_END_FACTORY_REG(StridedSliceAssignV2)
} // namespace ge
#endif