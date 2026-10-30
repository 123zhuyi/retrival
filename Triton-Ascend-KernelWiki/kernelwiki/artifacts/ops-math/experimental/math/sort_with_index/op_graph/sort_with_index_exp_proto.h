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
 * \file sort_with_index_exp_proto.h
 * \brief SortWithIndex op prototype (graph mode). Interface aligned with truth source
 *        math/sort_with_index. dtype scope follows spec v1.2 / D1=1A: value {float16, float,
 *        bfloat16, int32}, index {int32}; sorted_index dtype follows index dtype (currently int32).
 *        910B-only experimental op; int64 index is not supported by the framework on non-RegBase
 *        (DAV_2201), so the proto is narrowed to int32 to keep the declaration consistent with
 *        def.cpp / spec v1.2 / README (declaration must not exceed implementation).
 */
#ifndef OPS_MATH_EXPERIMENTAL_SORT_WITH_INDEX_EXP_PROTO_H_
#define OPS_MATH_EXPERIMENTAL_SORT_WITH_INDEX_EXP_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief sort the input tensor togather with input index, return the value and it's index.
 *
 *@par Inputs:
 *Inputs include:
 * x: A Tensor, data ready for sort. Dtype support: float16, float, bfloat16, int32, support format: [ND].
 * index: A Tensor, indices of input x. Dtype support: int32, support format: [ND].
 *
 *@par Attributes:
 * @li axis: An optional attribute indicates the sorting axis. Defaults to "-1".
 * @li descending: An optional attribute indicates descending sort or not. Defaults to "False".
 * @li stable: An optional attribute indicates the sort result of sorted_index is stable or not. Defaults to "False".
 *
 *@par Outputs:
 * @li y: A Tensor, sorted value. Must have the same type as x, support format: [ND].
 * @li sorted_index: sorted indices permuted by sorted x's indices. Dtype must be the same as index, format: [ND].
 *
 *@par Restrictions:
 * Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 *
 *@attention Constraints:
 * The shapes of input x and index must be the same. Only last-axis sort is supported.
 */
REG_OP(SortWithIndex)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .INPUT(index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OUTPUT(sorted_index, TensorType({DT_INT32}))
    .ATTR(axis, Int, -1)
    .ATTR(descending, Bool, false)
    .ATTR(stable, Bool, false)
    .OP_END_FACTORY_REG(SortWithIndex)

} // namespace ge

#endif // OPS_MATH_EXPERIMENTAL_SORT_WITH_INDEX_EXP_PROTO_H_
