/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file one_hot_proto.h
 * \brief
 */

#ifndef ONE_HOT_PROTO_H_
#define ONE_HOT_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Returns a one-hot tensor. The locations represented by index in "x" take value "on_value",
*         while all other locations take value "off_value" .

* @par Inputs:
* Four inputs, including:
* @li x: A 1-7D tensor of indices, format supports ND, and data type must be one of the following types: int32, uint8, int64.
* @li depth: A scalar which is the depth of the one hot dimension, format supports ND, and data type must be int32 or int64
*     Its shape can be 1-8D, but only the first element make sense.
* @li on_value: A scalar. The value to fill in output when indices[j] = i, format supports ND.
*     Must be one of the following types: float16, float32, int64, int32, int8, uint8.
*     Its shape can be 1-8D, but only the first element make sense.
* @li off_value: A scalar. The value to fill in output when indices[j] != i, format supports ND.
*     Has the same type as "on_value". Its shape can be 1-8D, but only the first element make sense.

* @par Attributes:
* axis: The axis to fill. An int with a minimum value of -1 and a maximum value of dims of x. Defaults to "-1"

* @par Outputs:
* y: A 1-8D tensor. Has the same type as "on_value" . \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator OneHot.
*/
REG_OP(OneHot)
    .INPUT(x, TensorType({DT_UINT8, DT_INT32, DT_INT64}))
    .INPUT(depth, TensorType({DT_INT32, DT_INT64}))
    .INPUT(on_value, TensorType::BasicType())
    .INPUT(off_value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(OneHot)
} // namespace ge

#endif
