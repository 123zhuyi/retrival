/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logical_not_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LOGICAL_NOT_H_
#define OPS_OP_PROTO_INC_LOGICAL_NOT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns the truth value of NOT "x" element-wise.

*@par Inputs:
*x: A ND Tensor of type bool. \n

*@par Outputs:
*y: A ND Tensor of type bool. \n

*@attention Constraints:
* The input and output values are "1" or "0", corresponding to bool values "true" and "false". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator logical_not.
*/
REG_OP(LogicalNot)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalNot)

} // namespace ge

#endif // OPS_OP_PROTO_INC_LOGICAL_NOT_H_

