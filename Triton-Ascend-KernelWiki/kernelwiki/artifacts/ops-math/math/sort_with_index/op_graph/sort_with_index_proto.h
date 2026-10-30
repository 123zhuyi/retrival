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
 * \file sort_with_index_proto.h
 * \brief
 */
#ifndef OPS_MATH_SORT_WITH_INDEX_GRAPH_PLUGIN_SORT_WITH_INDEX_PROTO_H_
#define OPS_MATH_SORT_WITH_INDEX_GRAPH_PLUGIN_SORT_WITH_INDEX_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief sort the input tensor togather with input index, return the value and it's index.
*
*@par Inputs:
*Inputs include:
* x: A Tensor, data reday for sort. Dtype support: float16, float, bfloat16, int16, int8, int32, int64, uint8, uint32,
        uint16, uint64, support format: [ND].
* index: A Tensor, indices of input x. Dtype support: int32, support format: [ND].
*
*@par Attributes:
* @li axis: An optional attribute indicates the sorting axis. Defaults to "-1".
* @li descending: An optional attribute indicates desending sort or not. Defaults to "False".
* @li stable: An optional attribute indicates the sort result of y2 is stable or not. Defaults to "False".
*
*@par Outputs:
* @li y: A Tensor, sorted value. Must have the same type as x, support format: [ND].
* @li sorted_index: sorted indices premuted by sorted x's indices. Dtype must be int32, support format: [ND].
*
*@par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*
*@attention Constraints:
* The shapes of input x and index must be the same.
*/
REG_OP(SortWithIndex)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(sorted_index, TensorType({DT_INT32, DT_INT64}))
    .ATTR(axis, Int, -1)
    .ATTR(descending, Bool, false)
    .ATTR(stable, Bool, false)
    .OP_END_FACTORY_REG(SortWithIndex)

} // namespace ge

#endif // OPS_MATH_SORT_WITH_INDEX_GRAPH_PLUGIN_SORT_WITH_INDEX_PROTO_H_

