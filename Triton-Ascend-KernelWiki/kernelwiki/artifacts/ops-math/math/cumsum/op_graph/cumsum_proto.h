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
 * \file cumsum_proto.h
 * \brief
 */
#ifndef CUMSUM_PROTO_H_
#define CUMSUM_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the cumulative sum of the tensor "x" along "axis" .

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types:
* int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32,
* double, complex64, complex128, bfloat16.
* @li axis: A Tensor of type int32 or int64. Range is [-rank(x),rank(x)). Dim and shape must be 1.
*
* @par Attributes:
* @li exclusive: A bool. Defaults to "False". If "False", performs inclusive cumsum, which means that the first element
* of the input is identical to the first element of the output. If "True", performs exclusive cumsum.
* @li reverse: A bool. Defaults to "False". If "True", the cumulative sum is calculated from the end of the 
* tensor towards the beginning. If "False", the cumulative sum is calculated from the beginning of the tensor towards
* the end.
*
* @par Outputs:
* y: A Tensor. Has the same type and shape as "x".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Cumsum.
*/
REG_OP(Cumsum)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(Cumsum)

} // namespace ge

#endif 

