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
 * \file masked_fill_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Replace the value of X with value according to mask.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of dtype is bfloat16 or float16 or float32 or int64 or int32 or int8 or bool.
* @li mask: A Tensor of dtype bool.
* @li value: A Tensor of dtype bfloat16 or float16 or float32 or int64 or int32 or int8 or bool. \n

* @par Outputs:
* y: A tensor. Must be one of the following dtypes:
* bfloat16, float16, float32, int64, int32, int8, bool.

* @attention Constraints:
* @li The input tensors of x and mask must meet the broadcast relationship.
* @li The dtype of value must be converted to the dtype of x.
* @li The shape of y is formed by broadcasting x, masked and value.
*/
REG_OP(MaskedFill)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64, DT_BOOL}))
    .INPUT(mask, TensorType({DT_BOOL}))
    .INPUT(value, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64, DT_BOOL}))
    .OP_END_FACTORY_REG(MaskedFill)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_