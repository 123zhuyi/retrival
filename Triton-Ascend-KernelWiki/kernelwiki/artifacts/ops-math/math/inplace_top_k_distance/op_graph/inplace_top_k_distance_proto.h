/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_TOP_K_DISTANCE_PROTO_H
#define INPLACE_TOP_K_DISTANCE_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief After a set of sorted data and a new set of data are re-sorted, get the first k data. \n
 *
 * @par Inputs:
 * Six inputs, including:
 * @li topk_pq_distance: A sorted Tensor, Will be updated after calculation.
 * Must be one of the following types: float32, float16.
 * @li topk_pq_index: A Tensor of type int32, index corresponding to topk_pq_distance.
 * @li topk_pq_ivf: A Tensor of type int32 , the bucket number corresponding to topk_pq_distance.
 * @li pq_distance: A Tensor of type float32 or float16,
 * the new data set will be reordered with topk_pq_distance and updated to topk_pq_distance.
 * @li pq_index: A Tensor of type int32, index corresponding to pq_distance.
 * @li pq_ivf: A scalar of type int32 , the bucket number corresponding to pq_distance. \n
 *
 * @par Attributes:
 * @li order: A string, indicates the sorting method of topk_pq_distance. \n
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
 */
REG_OP(InplaceTopKDistance)
    .INPUT(topk_pq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(topk_pq_index, TensorType({DT_INT32}))
    .INPUT(topk_pq_ivf, TensorType({DT_INT32}))
    .INPUT(pq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(pq_index, TensorType({DT_INT32}))
    .INPUT(pq_ivf, TensorType({DT_INT32}))
    .ATTR(order, String, "asc")
    .OP_END_FACTORY_REG(InplaceTopKDistance)
} // namespace ge

#endif
