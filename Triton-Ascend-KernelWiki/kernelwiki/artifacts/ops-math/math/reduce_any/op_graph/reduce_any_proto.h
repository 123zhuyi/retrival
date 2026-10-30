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
 * \file reduce_any_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_REDUCE_ANY_H_
#define OPS_OP_PROTO_INC_REDUCE_ANY_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* If "axes" is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:
* Only support bool,fp32
*
*@par Inputs:
*@li x : The tensor to reduce.
*@li axes: The int tensor, The dimensions to reduce.
*          If "None" (default), reduces all dimensions.
*          Must be in the range "[-rank(x), rank(x))".
*@par Attributes:
* keep_dims: bool, default false.
*If true, retains reduced dimensions with length 1.
*
*@par Outputs:
* y: The reduced tensor
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator reduce_any.
*
*/
REG_OP(ReduceAny)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAny)
} // namespace ge

#endif // OPS_OP_PROTO_INC_REDUCE_ANY_H_

