/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPK_PROTO_H_
#define TOPK_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Finds the k largest or smallest values and indices along a dimension.
 *@par Inputs:
 *x: A Tensor of type float16/float/double/int8/int16/int32/int64/uint8/uint16/uint32/uint64/bfloat16. \n
 *k: A Tensor of type int32, specifying the number of top elements.
 *@par Attributes:
 *dim: An optional int, specifying the dimension along which to perform topk. Default: -1.
 *largest: An optional bool, specifying whether to select largest or smallest values. Default: true.
 *sorted: An optional bool, specifying whether to sort the output. Default: true.
 *@par Outputs:
 *values: A Tensor of same type as x, containing the k largest/smallest values.
 *indices: A Tensor of type int32, containing the indices of the k largest/smallest values.
 */
REG_OP(TopK)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(largest, Bool, true)
    .ATTR(dim, Int, -1)
    .OP_END_FACTORY_REG(TopK)
} // namespace ge

#endif // TOPK_PROTO_H_
