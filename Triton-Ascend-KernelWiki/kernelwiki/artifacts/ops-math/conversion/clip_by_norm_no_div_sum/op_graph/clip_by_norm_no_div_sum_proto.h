/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANNDEV_CLIP_BY_NORM_NO_DIV_SUM_PROTO_H
#define CANNDEV_CLIP_BY_NORM_NO_DIV_SUM_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Performs element-wise ClipByNormNoDivSum operation.
*       y = Max(Select(x <= greater_zeros, x, Sqrt(Select(x > greater_zeros, x, select_ones))), maximum_ones)
*       All inputs support NumPy-style broadcast.

*@par Inputs:
* @li x: An ND Tensor. Must be one of the following types: float16, float32.
* @li greater_zeros: An ND Tensor. Same dtype as x.
* @li select_ones: An ND Tensor. Same dtype as x.
* @li maximum_ones: An ND Tensor. Same dtype as x.

*@par Outputs:
* @li y: An ND Tensor. Same dtype and shape as broadcasted inputs.
*/
REG_OP(ClipByNormNoDivSum)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(greater_zeros, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(select_ones, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(maximum_ones, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(ClipByNormNoDivSum)

} // namespace ge
#endif // CANNDEV_CLIP_BY_NORM_NO_DIV_SUM_PROTO_H
