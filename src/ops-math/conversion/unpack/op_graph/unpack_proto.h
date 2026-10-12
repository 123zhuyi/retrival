/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Unpacks the given dimension of a rank-R Tensor "x" into rank-(R-1)
* tensors.

* @par Inputs:
* x: A rank-R tensor (R > 0) of type BasicType.(BasicType includes:
* complex128, complex64, double, float32, float16, int16, int32, int64, int8,
* qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
* bfloat16, complex32.) \n

* @par Attributes:
* @li num: A required int, specifying the number of tensors to be unpacked to.
* Defaults to "None".
* @li axis: An optional int, specifying the axis to unpack along. The value range
* is [-R, R). Defaults to "0". \n

* @par Outputs:
* y: Dynamic output. The list of Tensor objects unpacked from "x", of type BasicType . \n

* @attention Constraints:
* For the ND format, "axis" is in the range [-R, R). \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Unstack.
*/

REG_OP(Unpack)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num, Int)
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(Unpack)
}  // namespace ge

#endif