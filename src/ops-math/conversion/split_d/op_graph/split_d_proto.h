/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPLIT_D_COMBINATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPLIT_D_COMBINATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors .

* @par Inputs:
* One input:
* x:An ND Tensor.
* Must be one of the following types: float16, float32, int32, int8, int16,
  int64, uint8, uint16, uint32, uint64, bool, bfloat16. \n

* @par Attributes:
* @li split_dim: A required int includes all types of int.
  Specifies the dimension along which to split. No default value.
* @li num_split: A required int includes all types of int.
  Specifies the number of output tensors. No default value . \n

* @par Outputs:
* y:Dynamic output. A list of output tensors. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "num_split" is greater than or equals to 1.
* @li "num_split" is divisible by the size of dimension "split_dim".
* @li "split_dim" is in the range [-len(x.shape), (x.shape)-1] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Split. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Split instead.
*/
REG_OP(SplitD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                          DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                                   DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitD)
}  // namespace ge

#endif