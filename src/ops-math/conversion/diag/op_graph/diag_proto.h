/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file diag_proto.h
 * \brief
 */
#ifndef OPS_CONVERSION_DIAG_GRAPH_PLUGIN_DIAG_PROTO_H_
#define OPS_CONVERSION_DIAG_GRAPH_PLUGIN_DIAG_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Create a diagonal tensor

* @par Inputs:
* One input, include:
* x: A mutable Tensor with rank k, where k is at most 4. Must be one of the Must be one of the following types:
*     bfloat16, float16, float32, double, int32, int64, complex64, complex128. Supported format list ["ND"]. \n
*     Note: bfloat16 is only supported on Ascend950PR/Ascend950DT.

* @par Outputs:
* y: A mutable Tensor. Has the same type as "x". Supported format list ["ND"]. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Diag.
*/

REG_OP(Diag)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Diag)

} // namespace ge

#endif