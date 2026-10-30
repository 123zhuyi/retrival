/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_TOP_K_PQ_DISTANCE_H_
#define OPS_OP_PROTO_TOP_K_PQ_DISTANCE_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Finds values and indices of the "k" largest or least elements for the last dimension. \n
 *
 * @par Inputs:
 * Dynamin inputs, including:
 * @li actual_count: A Tensor of type int32, the actual number of pq_distance.
 * @li pq_distance: A Tensor, Will be updated after calculation. Must be one of the following types: float32, float16.
 * @li grouped_extreme_distance: A Tensor, the extremum in each group. Must be one of the following types: float32,
 * float16.
 * @li pq_index: A Tensor of type int32, index corresponding to pq_distance.
 * @li pq_ivf: A Tensor of type int32 , the bucket number corresponding to pq_distance.
 *
 * @par Attributes:
 * @li order: A string, indicates the sorting method of topk_pq_distance. \n
 * @li k: Int, k maximum or minimum values. \n
 * @li group_size: Int, the group size of the extremum. \n
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
 */
REG_OP(TopKPQDistance)
    .DYNAMIC_INPUT(actual_count, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(pq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_INPUT(grouped_extreme_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_INPUT(pq_ivf, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(pq_index, TensorType({DT_INT32}))
    .OUTPUT(topk_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(topk_ivf, TensorType({DT_INT32}))
    .OUTPUT(topk_index, TensorType({DT_INT32}))
    .ATTR(order, String, "ASC")
    .REQUIRED_ATTR(k, Int)
    .REQUIRED_ATTR(group_size, Int)
    .OP_END_FACTORY_REG(TopKPQDistance)
} // namespace ge

#endif // OPS_OP_PROTO_TOP_K_PQ_DISTANCE_H_
