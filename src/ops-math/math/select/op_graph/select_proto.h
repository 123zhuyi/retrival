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
 * \file select_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SELECT_H_
#define OPS_OP_PROTO_INC_SELECT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Selects elements from "x1" or "x2", depending on "condition" . \n

* @par Inputs:
* Three inputs, including:
* @li condition: A tensor of type bool. Select x1 or x2 depending on this condition,
 * when condition is true, return x1, otherwise return x2.
* @li x1: A tensor. Must be one of the following types: bfloat16, float16, float32,
 * int32, int8, uint8, int16, uint16, double, complex64, int64, complex128, bool,
 * qint8, quint8, qint16, quint16, qint32, uint32, uint64, string.
 * Format:ND
* @li x2: A tensor of the same type, size, and shape as "x1". Format:ND

* @par Outputs:
* y: A Tensor. Has the same type as "x1". format:ND

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Select.
*/
REG_OP(Select)
    .INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(x1, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .INPUT(x2, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OP_END_FACTORY_REG(Select)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SELECT_H_

