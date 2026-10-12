/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_BIAS_H_
#define OPS_OP_PROTO_INC_BIAS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Add 'bias' to 'x'.
 *
 * @par Inputs:
 * Two inputs, including:
 * @li x: An ND tensor of type bfloat16, float16 or float32.
 * @li bias: An ND tensor of type bfloat16, float16 or float32. Shape rule see attention Constraints.
 *
 * @par Attributes:
 * @li axis: An optional int32 used to compute the shape of bias input from the online bottoms. Defaults to "1".
 * @li num_axes: An optional int32 used to compute the shape of
 * bias input from a Caffe model trained offline. Defaults to "1".
 * @li bias_from_blob: An optional bool. If "true", bias is input from a Caffe model trained offline.
 * If "false", bias is input from online bottoms. Defaults to "true".
 *
 * @par Outputs:
 * y: An ND tensor of type bfloat16, float16 or float32.
 *
 * @attention Constraints:
 * Assume that the shape length of "x" is "n" and that of "bias" is "m".
 * @li "axis" is within the range [-n, n-1]. num_axes >= -1.
 * @li If "bias_from_blob = true", "num_axes = -1", and "axis >= 0",
 * the ith axis of "bias" and the (i+"axis")th axis of "x" must have the same size (0 <= i < n-axis).
 * If "axis < 0", the ith axis of "bias" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < -axis).
 * @li If "bias_from_blob = true" and "num_axes = 0", "bias" is a scalar with shape length 1 and dimension size 1.
 * @li If "bias_from_blob = true", "num_axes > 0, and "axis >= 0",
 * "axis + num_axes" must be less than or equal to "n" and the ith axis of "bias" and
 * the (i+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
 * If "axis < 0", "n + axis + num_axes" must be less than or equal to "n" and
 * the ith axis of "bias" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
 * @li If "bias_from_blob = false", "bias" is not a scalar, and "axis >= 0",
 * "axis + m" must be less than or equal to "n" and the ith axis of "bias" and
 * the (i+"axis")th axis of "x" must have the same size (0 <= i < m).
 * If "axis < 0", "n + axis + m" must be less than or equal to "n" and
 * the ith axis of "bias" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < m). \n
 *
 * @par Third-party framework compatibility
 * Compatible with the Caffe operator Bias.
 */
#ifndef OPS_PROTO_DEF_BIAS
#define OPS_PROTO_DEF_BIAS
REG_OP(Bias)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(axis, Int, 1)
    .ATTR(num_axes, Int, 1)
    .ATTR(bias_from_blob, Bool, true)
    .OP_END_FACTORY_REG(Bias)
#endif
} // namespace ge

#endif // OPS_OP_PROTO_INC_BIAS_H_
