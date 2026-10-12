/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANNDEV_ASSIGN_PROTO_H
#define CANNDEV_ASSIGN_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Updates "ref" by assigning "value" to it. Donot support broadcasting operations.

*@par Inputs:
* @li ref: A ND Tensor. Support 1D ~ 8D. Must be one of the following types: bfloat16, float16, float32,
*    double, int8, int16, int32, int64, uint8, uint16, uint32, uint64, complex32,
*    complex64, complex128, qint8, quint8, qint16, qint32, quint16, bool, string.
*    Support format list: ["NC1HWC0", "ND", "C1HWNCoC0", "FRACTAL_Z", "FRACTAL_Z_3D",
*    "NDC1HWC0", "FRACTAL_NZ"].
*@li value: A ND Tensor of the same shape and dtype and format as "ref". \n

*@par Attributes:
*@li validate_shape: An optional bool. Defaults to "true".
                     If "true", the operation will validate that the shape of "value" matches the shape of the Tensor being assigned to.
*                    If "false", "ref" will take on the shape of "value".
*                    This attribute is reserved.
*@li use_locking: An optional bool. Defaults to false.
                  If True, the assignment will be protected by a lock;
                  otherwise the behavior is undefined, but may exhibit less contention.
*                 This attribute is reserved. \n

*@par Outputs:
*ref: A ND Tensor that holds the new value of ref after the value has been assigned.
*Has the same shape and dtype and format as the input "ref". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Assign.
*/
REG_OP(Assign)
    .INPUT(ref, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .INPUT(value, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(ref, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(validate_shape, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(Assign)

} // namespace ge
#endif // CANNDEV_ASSIGN_PROTO_H
