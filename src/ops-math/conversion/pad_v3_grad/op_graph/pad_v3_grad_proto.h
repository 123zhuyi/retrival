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
 * \file pad_v3_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PAD_V3_GRAD_OPS_H_
#define OPS_OP_PROTO_INC_PAD_V3_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Cal the grad of Pads.

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, bfloat16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64.
* @li paddings: A Tensor of type int32 or int64.
      1-D. The size is twice of the dimensionality of input x.

* @par Attributes:
* @li mode: An optional string, Defaults to "reflect", indicates paddings mode,
*     support "reflect", "edge", "constant", "symmetric", "circular".
* @li paddings_contiguous: An optional bool value, Defaults to true.
*     If true, paddings is arranged as [[begin0, end0], [begin1, end1], ...]
*     If false, paddings is arranged as [[begin0, begin1], ..., [end0, end1], ...]

* @par Outputs:
* y: A Tensor of the same type and dimensionality as "x". The shape of y and input x satisfy the
     mathematical relationship of the padding operation.

* @par Third-party framework compatibility:
* Compatible with ONNX operator PadGrad.
*/

REG_OP(PadV3Grad)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(mode, String, "reflect")
    .ATTR(paddings_contiguous, Bool, true)
    .OP_END_FACTORY_REG(PadV3Grad)

} // namespace ge

#endif
