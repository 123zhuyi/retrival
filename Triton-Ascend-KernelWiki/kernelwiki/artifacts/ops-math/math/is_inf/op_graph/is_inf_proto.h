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
 * \file is_inf_proto.h
 * \brief
 */
#ifndef OP_PROTO_IS_INF_PROTO_H_
#define OP_PROTO_IS_INF_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
namespace ge {
/**
 *@brief Compute element-wise infiniteness, return a boolean tensor.

 *@par Inputs:
 *x: A Tensor of type float16, float32, double, bfloat16, format is ND.

 *@par Outputs:
 *y: A Tensor. Has the same shape as x. Returns which elements of x are isinf, format is ND, dtype is bool.

 *@par Third-party framework compatibility.
 *Compatible with tensorflow IsInf operator.
 */
REG_OP(IsInf)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsInf)

}  // namespace ge


#endif  // OP_PROTO_IS_INF_PROTO_H_