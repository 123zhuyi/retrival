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
 * \file cos_proto.h
 * \brief
 */

#ifndef OPS_MATH_COS_GRAPH_PLUGIN_COS_PROTO_H_
#define OPS_MATH_COS_GRAPH_PLUGIN_COS_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes cosine of "x" element-wise.

* @par Inputs:
* x: A ND Tensor of type bfloat16, float16, float32, double, complex64, complex128.
* the format can be [NCHW,NHWC,ND]

* @par Outputs:
* y: A ND Tensor of the same dtype as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Cos. \n

*/
REG_OP(Cos)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Cos)

} // namespace ge

#endif // OPS_MATH_COS_GRAPH_PLUGIN_COS_PROTO_H_

