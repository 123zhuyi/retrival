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
 * \file grouped_bias_add_grad_proto.h
 * \brief
 */
#ifndef OP_PROTO_GROUPED_BIAS_ADD_GRAD_PROTO_H_
#define OP_PROTO_GROUPED_BIAS_ADD_GRAD_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Backwards calculation of GroupedBiasAdd.
* @par Inputs:
* @li grad_y: A Tensor. Type is:BFloat16, Float16 or Float32. When group_idx is inputted, the shape only supports 2 dimensions.
*             When group_idx is not inputted, the shape only supports 3 dimensions, supports non-continuous tensors, and the data format supports ND.
* @li group_idx: An optional Tensor. Type is:Int32 or Int64.
*             Optional parameter, the end position of each group, shape only supports 1 dimension, supports non-continuous tensors, and the data format supports ND.
* @par Outputs:
* @li grad_bias: A Tensor. Data type must match that of 'grad_y' (BFloat16, Float16 or Float32).
*             - Shape constraints are determined based on the dimensionality of 'grad_y':
*               - If 'grad_y' has 3 dimensions ('group_idx' is not provided), then:
*                    'grad_bias.shape = [grad_y.shape[0], grad_y.shape[2]]'
*               - If 'grad_y' has 2 dimensions ('group_idx' is provided), then:
*                    'grad_bias.shape = [group_idx.shape[0], grad_y.shape[1]]'                
*             - Supports non-continuous tensors, and data format supports "ND".

* @par Attributes:
* @li group_idx_type: An optional Int, specifying the significance of group_idx, default to 0.
* @attention Constraints:
* group_idx: A maximum of 2048 groups are supported. \n
* When group_idx is inputted, it is required to ensure that the values of the tensor do not exceed the maxium value of INT32 and are non-negative. \n
* When group_idx is inputted and group_idx_type is 0, it is necessary to ensure that the tensor data is in ascending order, and the last numerical value is equal to the
* size of the 0th dimension of grad_y. \n
* When group_idx is inputted and group_idx_type is 1, it is necessary to ensure that the sum of the tensor values must equal the size of grad_y in the 0th dimension.
*/
REG_OP(GroupedBiasAddGrad)
    .INPUT(grad_y, "T")
    .OPTIONAL_INPUT(group_idx, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(grad_bias, "T")
    .ATTR(group_idx_type, Int, 0)
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GroupedBiasAddGrad)

}  // namespace ge


#endif  // OP_PROTO_GROUPED_BIAS_ADD_GRAD_PROTO_H_