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
 * \file reduce_all_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_REDUCE_ALL_H_
#define OPS_OP_PROTO_INC_REDUCE_ALL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension .

*@attention Constraints:
* Support bool, fp32, fp16, bf16

*@par Inputs:
*Two inputs, including:
*@li x: The tensor to reduce.
*@li axis: A mutable Tensor with int dtype, The dimensions to reduce.
*If None, reduces all dimensions.
*Must be in the range [- rank (input_sensor), rank (input_sensor)) .

*@par Attributes:
*keep_dims: A bool, default false.
*If true, retains reduced dimensions with length 1 .

*@par Outputs:
*y: The reduced tensor .

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceAll.
*/
REG_OP(ReduceAll)
    .INPUT(x, TensorType({DT_BOOL, DT_BF16, DT_FLOAT, DT_FLOAT16}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAll)

} // namespace ge

#endif // OPS_OP_PROTO_INC_REDUCE_ALL_H_

