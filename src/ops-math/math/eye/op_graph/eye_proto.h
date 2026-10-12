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
 * \file eye_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_EYE_H_
#define OPS_OP_PROTO_INC_EYE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

* @par Inputs:
* No inputs

* @par Attributes:
* @li num_rows: An required int. \n
* @li num_columns: An optional int.Defaults to 0. \n
* @li batch_shape: An optional ListInt.Defaults to []. \n
* @li dtype: An optional int.Defaults to 0. \n

* @par Outputs:
* y: A Tensor with targeted type and shape. Must be one of the following types:
*   complex128, complex64, double, float32, float16, int16, int32, int64, int8, qint16,
*   qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8, bfloat16, complex32, bool. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Eye. \n
*/
REG_OP(Eye)
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))    /* "Result, has targeted element type" */
    .REQUIRED_ATTR(num_rows, Int)
    .ATTR(num_columns, Int, 0)
    .ATTR(batch_shape, ListInt, {})
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(Eye)

} // namespace ge

#endif // OPS_OP_PROTO_INC_EYE_H_

