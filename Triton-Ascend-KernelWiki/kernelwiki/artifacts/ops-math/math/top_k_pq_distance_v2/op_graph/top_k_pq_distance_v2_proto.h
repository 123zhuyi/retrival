/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOP_K_PQ_DISTANCE_V2_PROTO_H
#define TOP_K_PQ_DISTANCE_V2_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Finds values and indices of the "k" largest or least elements for the last dimension. \n
 *
 * @par Inputs:
 * Two inputs, including:
 * @li pq_distance: A Tensor, Will be updated after calculation. Must be one of the following types: float32, float16,
 * int32.
 * @li grouped_extreme_distance: A Tensor, the extremum in each group. Must be one of the following types: float32,
 * float16, int32.
 *
 * @par Outputs:
 * Two outputs, including:
 * @li topk_distance: A Tensor, values of the "k" largest or least elements for the last dimension. Must be one of the
 * following types: float32, float16, int32.
 * @li topk_index: A Tensor, indices of the "k" largest or least elements for the last dimension. dtype is int32.
 *
 * @par Attributes:
 * @li order: A string, indicates the sorting method, Must be one of the following string: "ASC" or "DES". default
 * is "ASC".
 * @li k: Int, k maximum or minimum values, required.
 * @li group_size: Int, the group size of the extremum, required.
 *
 */
REG_OP(TopKPQDistanceV2)
    .INPUT(pq_distance, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(grouped_extreme_distance, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(topk_distance, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(topk_index, TensorType({DT_INT32}))
    .ATTR(order, String, "ASC")
    .REQUIRED_ATTR(k, Int)
    .REQUIRED_ATTR(group_size, Int)
    .OP_END_FACTORY_REG(TopKPQDistanceV2)
} // namespace ge

#endif
