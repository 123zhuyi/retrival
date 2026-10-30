/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file slice_last_dim_proto.h
 * \brief
 */
#ifndef OPS_OP_SLICE_LAST_DIM_PROTO_H_
#define OPS_OP_SLICE_LAST_DIM_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Slices the last dimension of a tensor from start to end with stride. \n

*@par Inputs:
*x: A tensor of type float16, float32, double, int8, int16, int32, int64. \n

*@par Outputs:
*y: A tensor with the same type as x. \n

*@par Attributes:
*@li start: Required. Start index of the last dimension.
*@li end: Required. End index of the last dimension.
*@li stride: Optional. Stride of slicing. Defaults to 1. \n

*@par Third-party framework compatibility
* No compatibility
*/
REG_OP(SliceLastDim)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(start, Int)
    .REQUIRED_ATTR(end, Int)
    .ATTR(stride, Int, 1)
    .OP_END_FACTORY_REG(SliceLastDim)
} // namespace ge
#endif
