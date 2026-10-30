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
 * \file diag_part_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_DIAG_PART_H_
#define OPS_OP_PROTO_INC_DIAG_PART_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns the batched diagonal part of a batched tensor . \n

* @par Inputs:
* x: A Tensor. Must be one of the following types:
*    float16, float32, int32, int64, double, complex64, complex128. Supported format list ["ND"]. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x". Supported format list ["ND"]. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator DiagPart.
*/
REG_OP(DiagPart)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(DiagPart)

} // namespace ge

#endif // OPS_OP_PROTO_INC_DIAG_PART_H_
