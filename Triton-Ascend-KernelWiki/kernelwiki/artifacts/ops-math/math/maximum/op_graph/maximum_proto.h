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
 * \file maximum_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MAXIMUM_H_
#define OPS_OP_PROTO_INC_MAXIMUM_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns the max of "x1" and "x2" (i.e. x1 > x2 ? x1: x2) element-wise. Support broadcasting operations. \n

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: float16, float32, double, int32, int64, bfloat16, int8,
uint8.
* @li x2: A ND Tensor of the same dtype as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Maximum.
*/
REG_OP(Maximum)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .OP_END_FACTORY_REG(Maximum)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MAXIMUM_H_