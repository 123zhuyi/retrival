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
 * \file real_div_proto.h
 * \brief
 */
#ifndef OPS_OP_REAL_DIV_PROTO_H_
#define OPS_OP_REAL_DIV_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge
{
/**
* @brief Returns x1/x2 element-wise for real types. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor.
* Must be one of the following types: bfloat16, float16, float32, double, uint16,
* int8, uint8, int16, int32, int64, complex64, complex128, bool.
* @li x2: A ND Tensor.
* Must be one of the following types: bfloat16, float16, float32, double, uint16,
* int8, uint8, int16, int32, int64, complex64, complex128, bool. \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype and format as input "x1" if the type of "x1" is not bool.
     If the type of "x1" is bool, y is float type. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator RealDiv.
*/
REG_OP(RealDiv)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(RealDiv)
} // namespace ge
#endif