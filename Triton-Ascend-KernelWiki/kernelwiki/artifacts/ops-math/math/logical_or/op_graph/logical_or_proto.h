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
 * \file logical_or_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LOGICAL_OR_H_
#define OPS_OP_PROTO_INC_LOGICAL_OR_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns the truth value of x1 OR x2 element-wise. Support broadcasting operations.

*
*@par Inputs:
*@li x1: A ND tensor. Must be one of the following types: bool, int8, uint8, int16, int32, int64, float16, bfloat16, float32.
*@li x2: A ND tensor of the same dtype as "x1".
*
*@attention Constraints:
* LogicalOr supports broadcasting.
*
*@par Outputs:
* y: A ND tensor of type bool.
*
*@par Third-party framework compatibility
*Compatible with the PyTorch operator torch.logical_or.
*
*/
REG_OP(LogicalOr)
    .INPUT(x1, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalOr)

} // namespace ge

#endif // OPS_OP_PROTO_INC_LOGICAL_OR_H_

