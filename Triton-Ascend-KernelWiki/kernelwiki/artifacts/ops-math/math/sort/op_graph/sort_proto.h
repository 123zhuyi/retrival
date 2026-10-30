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
 * \file sort_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SORT_H_
#define OPS_OP_PROTO_INC_SORT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief sort the input tensor and return the value of index.
*
*@par Inputs:
*Inputs include:
* x: A Tensor. Supported type: float16, float32, int16, int8, uint8, int32, int64, bfloat16, uint32, uint16, uint64.
* Supported format: ND . \n

*@par Attributes:
* @li axis: An optional attribute indicates the sorting axis. Defaults to "-1".
* @li descending: An optional attribute indicates descending sort or not. Defaults to "false".
* @li stable: An optional attribute indicates the sort result of y2 is stable or not. Defaults to "false" . \n
* @li y2_dtype: An optional attribute indicates the sort result of y2's dtype. Defaults to "int32" . \n

*@par Outputs:
* @li y1: A Tensor. Must have the same type and format as x.
* @li y2: A Tensor. Indices of y1 in x. Dtype must be int32 or int64. Supported format: ND . \n

*@attention Constraints:
* The operator depends on the unstable sorting algorithm.
*/
REG_OP(Sort)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8,
                          DT_UINT8, DT_INT32, DT_INT64, DT_BF16, 
                          DT_UINT32, DT_UINT16, DT_UINT64}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8,
                            DT_UINT8, DT_INT32, DT_INT64, DT_BF16,
                            DT_UINT32, DT_UINT16, DT_UINT64}))
    .OUTPUT(y2, TensorType({DT_INT32, DT_INT64}))
    .ATTR(axis, Int, -1)
    .ATTR(descending, Bool, false)
    .ATTR(stable, Bool, false)
    .ATTR(y2_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(Sort)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SORT_H_

