/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SLICE_WRITE_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SLICE_WRITE_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief write tensor value to tensor x.
*@par Inputs:
*x: A Tensor of type float16/float/double/int32/int64. \n
*begin:A Tensor of type int32/int64. \n
*value: A Tensor of type float16/float/double/int32/int64.
*@par Outputs:
*x: same tensor with input x.
*/
REG_OP(SliceWrite)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(begin, TensorType({DT_INT32, DT_INT64}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(SliceWrite)
}  // namespace ge

#endif