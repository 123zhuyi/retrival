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
 * \file reduce_log_sum_exp_proto.h
 * \brief
 */

#ifndef REDUCE_LOG_SUM_EXP_PROTO_H_
#define REDUCE_LOG_SUM_EXP_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Computes the log and sum and exp of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float16, bfloat16.
* @li axes: A 1D list or tuple of int32 or int64. Specifies the dimensions to reduce. \n
*
* @par Attributes:
* keep_dims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false" . \n
*
* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x" . \n
*
* @par Third-party framework compatibility
* Compatible with the Onnx operator ReduceLogSumExp.
*/
REG_OP(ReduceLogSumExp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(axes, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceLogSumExp)
} // namespace ge

#endif