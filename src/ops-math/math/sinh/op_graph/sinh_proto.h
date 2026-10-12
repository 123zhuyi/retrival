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
 * \file sinh_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Computes hyperbolic sine of "x" element-wise.

* @par Inputs:
* x: An NCHW, NHWC,or ND Tensor that supports the data type UnaryDataType. \n

* @par Outputs:
* y: A ND Tensor with the same dtype and shape of input "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sinh. \n

*/
REG_OP(Sinh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sinh)
}

#endif  // OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
