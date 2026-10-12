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
 * \file bitwise_not_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_BITWISE_NOT_H_
#define OPS_OP_PROTO_INC_BITWISE_NOT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes bitwise NOT element-wise (y = ~x).
* For integer dtypes this is the bitwise complement (~x, i.e. -x-1 for signed);
* for BOOL it is the logical NOT (0<->1).

*@par Inputs:
* One input:
*x: An ND tensor. Support 0D~8D. Must be one of the following types:
* int8, int16, int32, int64, uint8, bool.

*@par Outputs:
*y: An ND Tensor. Has the same dtype and shape as input "x".

*@par Third-party framework compatibility
* Compatible with torch.bitwise_not / numpy.invert.
*/
REG_OP(BitwiseNot)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BOOL}))
    .OP_END_FACTORY_REG(BitwiseNot)

} // namespace ge

#endif // OPS_OP_PROTO_INC_BITWISE_NOT_H_
