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
 * \file moe_compute_expert_tokens_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_
#define OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Binary finds the position of the last row processed by each expert in the sorted_experts array.
 * @par Inputs:
 * @li sorted_experts: An 1D Tensor, sorted expert array. Type is:Int32. Shape should be less than 2^24.
 * @par Outputs:
 * @li total_rows_before_expert: A Tensor. Type is:Int32.
 * @par Attributes:
 * @li num_experts: Required parameter. Type is:Int. The value must be more than 0 and less than 2048.
 */
REG_OP(MoeComputeExpertTokens)
    .INPUT(sorted_experts, "T")
    .OUTPUT(total_rows_before_expert, "T")
    .REQUIRED_ATTR(num_experts, Int)
    .DATATYPE(T, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(MoeComputeExpertTokens)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_
