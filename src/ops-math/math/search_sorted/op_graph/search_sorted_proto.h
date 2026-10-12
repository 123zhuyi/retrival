/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RIGHT_SHIFT_PROTO_H
#define RIGHT_SHIFT_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief insert the values into the sorted sequence and return the index. \n

* @par Inputs:
* @li sorted_sequence: A Tensor of {DT_FLOAT16,DT_FLOAT,DT_INT16,DT_INT8,DT_UINT8,DT_INT32,DT_INT64},
                       the values of the last dim are sorted by ascending order.
* @li values: the inserted Tensor. Must have the same type as input. only the last dim can be different from
              the sorted_sequence.
* @li sorter:  if provided, a tensor matching the shape of the unsorted sorted_sequence containing a sequence of indices
               that sort it in the ascending order on the innermost dimension  \n

* @par Outputs:
* @li out: output tensor of the op, which is the same shape as input "values". Dtype is int32 or int64. \n

* @par Attributes:
* @li dtype: An optional type. Default value is DT_INT64, only supports DT_INT64/DT_INT32.

* @li right: An optional bool. Default value is false, false means the inserted position aligns to the left side when
             the sequence contains same value and the position candidates are not unique, while true means aligning to
             the right side when in such situation. \n

* @par Third-party framework compatibility
* Compatible with pytorch1.8.1 searchsorted operator.
*/

REG_OP(SearchSorted)
    .INPUT(sorted_sequence, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8,
                                DT_UINT8, DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8,
                                DT_UINT8, DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(sorter, TensorType({DT_INT64}))
    .OUTPUT(out, TensorType(DT_INT32, DT_INT64))
    .ATTR(dtype, Type, DT_INT64)
    .ATTR(right, Bool, false)
    .OP_END_FACTORY_REG(SearchSorted)
}

#endif