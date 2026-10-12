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
 * \file logical_and_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LOGICAL_AND_H_
#define OPS_OP_PROTO_INC_LOGICAL_AND_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns the truth value of x1 AND x2 element-wise. Support broadcasting operations.

*
*@par Inputs:
*@li x1: A tensor of type bool.
*@li x2: A tensor of the same dtype as "x1".
*
*@attention Constraints:
* LogicalAnd supports broadcasting.
*
*@par Outputs:
* y: A tensor of the same dtype as "x1".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogicalAnd.
*
*/
REG_OP(LogicalAnd)
    .INPUT(x1, TensorType({DT_BOOL}))
    .INPUT(x2, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalAnd)

} // namespace ge

#endif // OPS_OP_PROTO_INC_LOGICAL_AND_H_
