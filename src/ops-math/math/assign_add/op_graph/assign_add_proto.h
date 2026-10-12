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
 * \file assign_add_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ASSIGN_ADD_H_
#define OPS_OP_PROTO_INC_ASSIGN_ADD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Updates "ref" by adding "value" to it. Donot support broadcasting operations.

* @par Inputs:
* @li ref: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bfloat16, float16, float32, int8,
                     int16, int32, int64, uint8, uint16, uint32, uint64.
* @li value: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bfloat16, float16, float32, int8,
                     int16, int32, int64, uint8, uint16, uint32, uint64.

* @par Attributes:
* use_locking: An optional bool. Defaults to "False".
              If "True", the addition will be protected by a lock;
              otherwise the behavior is undefined, but may exhibit less contention.
*             This attribute is reserved.

* @par Outputs:
* ref: A ND Tensor that holds the new value of ref after the value has been added.

* @attention Constraints:
* An input tensor of type int64 must have a shape with size 1.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AssignAdd.
*/
REG_OP(AssignAdd)
    .INPUT(ref, TensorType::BasicType())
    .INPUT(value, TensorType::BasicType())
    .OUTPUT(ref, TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(AssignAdd)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ASSIGN_ADD_H_
