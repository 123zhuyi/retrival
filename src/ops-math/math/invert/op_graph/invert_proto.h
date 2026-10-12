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
 * \file invert_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_INVERT_H_
#define OPS_OP_PROTO_INC_INVERT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Reverses specific dimensions of a tensor.

* @par Inputs:
* One input: \n
* x: A ND Tensor, Must be one of the following types:
* int32, uint8, int16, int8, int64, uint16, uint32, uint64,
* and format can be [NCHW,NHWC,ND]. \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype and format as "x". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Invert.
*/
REG_OP(Invert)
    .INPUT(x, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(Invert)

} // namespace ge

#endif // OPS_OP_PROTO_INC_INVERT_H_
