/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOP_K_V2_D_PROTO_H_
#define TOP_K_V2_D_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Finds the k largest or smallest values and indices along a dimension.
 *@par Inputs:
 * Three inputs, including:
 *@li x: A Tensor of type float16/float/double/int8/int16/int32/int64/uint8/uint16/uint32/uint64. \n
 *@li k: A Tensor of type int32, specifying the number of top elements.
 *@li assist_seq: A Tensor of type float16, assisting in sequence computation. \n
 *@par Attributes:
 *@li sorted: An optional bool, specifying whether to sort the output. Default: true.
 *@li dim: An optional int, specifying the dimension along which to perform topk. Default: -1.
 *@li largest: An optional bool, specifying whether to select largest or smallest values. Default: true.
 *@par Outputs:
 *@li values: A Tensor of same type as x, containing the k largest/smallest values.
 *@li indices: A Tensor of type int32, containing the indices of the k largest/smallest values.
 */
REG_OP(TopKV2D)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(assist_seq, TensorType({DT_FLOAT16}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(dim, Int, -1)
    .ATTR(largest, Bool, true)
    .OP_END_FACTORY_REG(TopKV2D)
} // namespace ge

#endif // TOP_K_V2_D_PROTO_H_
