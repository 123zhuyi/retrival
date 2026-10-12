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
 * \file bias_add_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_BIAS_ADD_GRAD_H_
#define OPS_OP_PROTO_INC_BIAS_ADD_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Performs the the backward operation for "BiasAdd" on the "bias" tensor.
*        It accumulates all the values from out_backprop into the feature
*        dimension. For NHWC data format, the feature dimension is the last.
*        For NCHW data format, the feature dimension is the third-to-last .

* @par Inputs:
* x: A Tensor of type NumberType . \n

* @par Attributes:
* data_format: A required attr. Data format. Defaults to "NHWC" . \n

* @par Outputs:
* y: A Tensor.Has the same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BiasAddGrad.
*/
REG_OP(BiasAddGrad)
    .INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(BiasAddGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_BIAS_ADD_GRAD_H_
