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
 * \file top_k_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_TOP_K_V2_H_
#define OPS_OP_PROTO_INC_TOP_K_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Finds values and indices of the "k" largest elements for the last
* dimension . \n

* @par Inputs:
* Two inputs, including:
* @li x: A 1D-8D tensor, with the last dimension at least "k".
* Supported type: float16, float32, int16, int8, uint8, int32, int64, bfloat16, uint32, uint16, uint64. 
* Supported format: ND.
* @li k: A 0D Tensor. Supported type: int32, int64. 
* Supported format: ND.
* Number of top elements to look for along the last dimension (along each row
* for matrices) . \n

* @par Attributes:
* @li sorted: An optional bool. Defaults to "True".
* If "True", the returned "k" elements are themselves sorted.
* If "False", the returned "k" elements are not sorted.
* @li dim: An optional int. Defaults to -1. For reserved use.
* @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to true.
* If "True", the "k" largest elements are returned in descending order.
* If "False", the "k" smallest elements are returned in ascending order. \n
* @li indices_dtype: An optional attribute indicates the sort result of indices' dtype, either "DT_INT32(3)" or "DT_INT64(9)". Defaults to "DT_INT32(3)". \n

* @par Outputs:
* @li values: A Tensor, specifying the sorted data. Has the same type and format as
* "input".
* @li indices: A Tensor. Indices of values in x. Dtype must be "int32" or "int64". Supported format: ND . \n

* @see TopK()
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator TopKV2.
*/
REG_OP(TopKV2)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType::IndexNumberType())
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType::IndexNumberType())
    .ATTR(sorted, Bool, true)
    .ATTR(dim, Int, -1)
    .ATTR(largest, Bool, true)
    .ATTR(indices_dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(TopKV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_TOP_K_V2_H_

