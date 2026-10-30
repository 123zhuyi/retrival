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
 * \file acos_proto.h
 * \brief
 */
#ifndef ACOS_PROTO_H_
#define ACOS_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Computes acos of x element-wise.

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, bfloat16, float32,
*     double, int32, int64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same dtype as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Acos.
*
*/
REG_OP(Acos)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                             DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Acos)
} // namespace ge
#endif // ACOS_PROTO_H_