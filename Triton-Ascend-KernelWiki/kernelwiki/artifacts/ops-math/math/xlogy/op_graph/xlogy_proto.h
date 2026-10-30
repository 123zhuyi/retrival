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
 * \file xlogy_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_XLOGY_H_
#define OPS_OP_PROTO_INC_XLOGY_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes x1 * log(x2) element-wise. Returns 0 if x1 == 0, propagates NaN from x2.
*@par Inputs:
*Two inputs, including:
* @li x1: An ND Tensor. Must be one of: float16, float32, bfloat16, float64, complex64, complex128.
* @li x2: An ND Tensor. Must be one of: float16, float32, bfloat16, float64, complex64, complex128.
*
*@par Outputs:
* @li y: An ND Tensor with broadcast shape of x1 and x2.
*@par Third-party framework compatibility
*Compatible with TensorFlow/PyTorch xlogy.
*/
REG_OP(Xlogy)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Xlogy)

} // namespace ge

#endif // OPS_OP_PROTO_INC_XLOGY_H_
