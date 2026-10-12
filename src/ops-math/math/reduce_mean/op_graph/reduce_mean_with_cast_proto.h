/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_mean_proto.h
 * \brief
 */
#ifndef REDUCE_MEAN_WITH_CAST_PROTO_H_
#define REDUCE_MEAN_WITH_CAST_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Insert a Cast node for the ReduceMean operator . \n

* @par Inputs:
* Two inputs, including:
*  @li x: A ND Tensor. Must be one of the following types: float16, float32, int8, uint8.
*  @li axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType.
*    - If None (the default), reduces all dimensions.
*    - Must be in the range [-rank(x), rank(x)) . \n

* @par Attributes:
* keep_dims: A bool or NoneType.
*  - If true, retains reduced dimensions with length 1.
*  - If false, the rank of the tensor is reduced by 1 for each entry in axis.
* noop_with_empty_axes: A bool.
*  - If true, when axes = [], not reduce.
*  - If false, when axes = [], reduce all.
* dtype: enum.
*  - optional attr, could be one of the following types: DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8.
* @par Outputs:
* y: A ND Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator ReduceMeanWithCast.
*/
REG_OP(ReduceMeanWithCast)
    .INPUT(x, "T1")
    .INPUT(axes, "T2")
    .OUTPUT(y, "T3")
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .ATTR(dtype, Type, DT_UNDEFINED)
    .DATATYPE(T1, TensorType::NumberType())
    .DATATYPE(T2, TensorType::IndexNumberType())
    .DATATYPE(T3, TensorType::NumberType())
    .OP_END_FACTORY_REG(ReduceMeanWithCast)

} // namespace ge

#endif // REDUCE_MEAN_WITH_CAST_PROTO_H_