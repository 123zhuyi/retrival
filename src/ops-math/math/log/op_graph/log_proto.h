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
 * \file  log_proto.h
 * \brief
 */
#ifndef OPS_MATH_MATH_LOG_PROTO_H_
#define OPS_MATH_MATH_LOG_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes logarithm of x element-wise.
* y = log_base(shift + scale * x), with "base" > 0.

* @par Inputs:
* x: A ND Tensor of type uint8, int8, int16, int32, int64, float64,
*    float16, bfloat16, float32, bool, complex128 or complex64. \n

* @par Attributes:
* @li base: An optional float32, specifying the base "e". Defaults to "-1.0"

* @li scale: An optional float32, specifying the scale of input "x". Defaults
* to "1.0"
* @li shift: An optional float32, specifying the shift. Defaults to "0.0"

* @par Outputs:
* y: A tensor, when the input is of integer type, the y type is float32.
*    Other case, y has same type as "x". \n

* @attention Constraints:
* @li "base" is supposed to be greater than 0. Retaining the default
* value "-1" sets "base" to "e".
* @li If the input value of operator Log is within the range (0, 0.01] or
* [0.95, 1.05], the output accuracy is subject to change. \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Log.
* @li Compatible with the Caffe operator Log.
*/
REG_OP(Log)
    .INPUT(
        x, TensorType(
               {DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_BOOL,
                DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType::UnaryDataType())
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Log)

} // namespace ge

#endif // OPS_MATH_MATH_LOG_PROTO_H_
