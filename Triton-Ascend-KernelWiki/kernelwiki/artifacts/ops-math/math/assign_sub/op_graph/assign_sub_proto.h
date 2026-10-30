/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_ASSIGN_SUB_H_
#define OPS_OP_PROTO_INC_ASSIGN_SUB_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Updates "var" by subtracting "value" from it.\n
* This operation outputs "var" after the update is done. \n
* This makes it easier to chain operations that need to use the reset value.

*
* @par Inputs:
* @li var: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bfloat16, float32, float64, int32,
uint8 \n
* int16, int8, complex64, int64, qint8, quint8, qint32, uint16, complex128, uint32, uint64
* @li value: A tensor of the same dtype as "var".
*
* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True", the subtraction will be protected \n
* by a lock; otherwise the behavior is undefined, but may exhibit less contention.
*
* @par Outputs:
* y: A tensor. Has the same dtype as "var".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AssignSub.
*
*/
REG_OP(AssignSub)
    .INPUT(var, TensorType::NumberType())
    .INPUT(value, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(AssignSub)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ASSIGN_SUB_H_
