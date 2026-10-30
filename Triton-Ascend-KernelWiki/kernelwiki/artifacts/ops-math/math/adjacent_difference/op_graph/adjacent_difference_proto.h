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
 * \file adjacent_difference_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ADJACENT_DIFFERENCE_H_
#define OPS_OP_PROTO_INC_ADJACENT_DIFFERENCE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Compare two consecutive values in input tensor, if they are equal, return 0; otherwise, return 1. \n
* y[0] = 0;\n
* y[i] = x[i] == x[i - 1] ? 0 : 1;
*
*@par Inputs:
*Inputs include:
* x: A tensor. Dtype support: float16, float, bfloat16, int16, int8, int32, int64, uint8, uint32,
        uint16, uint64, support format: [ND].
*
*@par Outputs:
* y: A tensor. Must have the same shpae as x, dtype support int32, int64. Support format: [ND].
*
*@par Attributes:
* y_dtype: The output type, either "DT_INT32(3)" or "DT_INT64(9)". Defaults to "DT_INT32(3)".
*/
REG_OP(AdjacentDifference)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(y_dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(AdjacentDifference)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ADJACENT_DIFFERENCE_H_

