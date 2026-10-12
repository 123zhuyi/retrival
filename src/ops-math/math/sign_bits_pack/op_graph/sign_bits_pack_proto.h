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
 * \file sign_bits_pack_proto.h
 * \brief SignBitsPack 算子 GE IR 原型注册（图模式）
 *
 * 输入 x: DT_FLOAT/DT_FLOAT16，输出 y: DT_UINT8，属性 size (Int, 必选)。
 */
#ifndef OPS_OP_PROTO_INC_SIGN_BITS_PACK_H_
#define OPS_OP_PROTO_INC_SIGN_BITS_PACK_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief SignBitsPack: extract sign bits of float elements and pack into uint8 bytes.
 *   Each 8 sign bits are packed into 1 uint8 byte (MSB-first).
 *@par Inputs:
 * @li x: A 1D ND Tensor of type float16 or float32. Input float tensor.
 *
 *@par Outputs:
 * @li y: A 2D ND Tensor of type uint8, shape [size, ceil(N/8)/size]. Packed sign bits.
 *
 *@par Attributes:
 * @li size: Int, required. Output first dimension size, must be >= 1 and ceil(N/8) % size == 0.
 */
REG_OP(SignBitsPack)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .REQUIRED_ATTR(size, Int)
    .OP_END_FACTORY_REG(SignBitsPack)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SIGN_BITS_PACK_H_
