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
 * \file k_l_div_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_K_L_DIV_V2_H_
#define OPS_OP_PROTO_INC_K_L_DIV_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Kullback-Leibler divergence.

* @par Inputs:
* Two inputs, including:
* @li x: Tensor of arbitrary shape. Must be the type of following types: bfloat16, float16, float32.
* @li target: Tensor of the same shape and dtype as x. \n

* @par Attributes:
* @li reduction: An optional "string", Specifies the reduction to apply to the output;
* Reduction supports the modes of "sum", "mean", "none" and "batchmean", default value is "mean". \n
* @li log_target: An optional bool, a flag indicating whether target is passed in the log space. Default value is false. \n

* @par Outputs:
* y: A ND Tensor of the same dtype as x. When reduction is "none", input and output has same shape, otherwise output is a scalar.
* @par Third-party framework compatibility
* Compatible with the PyTorch operator kl_div_v2.
*/
REG_OP(KLDivV2)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .ATTR(log_target, Bool, false)
    .OP_END_FACTORY_REG(KLDivV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_K_L_DIV_V2_H_
