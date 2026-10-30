/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_COMPARE_AND_BIT_PACK_MATH_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_COMPARE_AND_BIT_PACK_MATH_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Compare values of input to threshold and pack resulting bits into
 a uint8.

 *@par Inputs:
 *The input size must be a non-negative int32 scalar Tensor. Inputs include:
 *@li x:Values to compare against threshold and bitpack.
 *@li threshold:Threshold to compare against. \n

 *@par Outputs:
 *y:The bitpacked comparisons. \n

 *@attention Constraints:
 *Currently, the innermost dimension of the tensor must be divisible by 8. \n

 *@par Third-party framework compatibility
 *Compatible with tensorflow CompareAndBitpack operator.
 */

REG_OP(CompareAndBitpack)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL}))
    .INPUT(threshold, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL}))
    .OUTPUT(y, TensorType(DT_UINT8))
    .OP_END_FACTORY_REG(CompareAndBitpack)
} // namespace ge

#endif
