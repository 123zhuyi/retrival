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
 * \file floor_div_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FLOOR_DIV_H_
#define OPS_OP_PROTO_INC_FLOOR_DIV_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Divides "x1/x2" element-wise, rounding toward the
*        most negative integer. Support broadcasting operations.

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor.
* Must be one of the following types: float16, float32, int32, int64, int8,
*     uint8, int16, uint16, double, bfloat16.
* @li x2: A ND Tensor of the same dtype as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FloorDiv.
*/
REG_OP(FloorDiv)
    .INPUT(
        x1, TensorType(
                {DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_BF16}))
    .INPUT(
        x2, TensorType(
                {DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_BF16}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(FloorDiv)

} // namespace ge

#endif // OPS_OP_PROTO_INC_FLOOR_DIV_H_
