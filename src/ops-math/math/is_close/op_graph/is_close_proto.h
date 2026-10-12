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
 * \file is_close_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_CLOSE_H_
#define OPS_OP_PROTO_INC_IS_CLOSE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns a new tensor with boolean elements representing \n
*if each element of input is “close” to the corresponding element of other

*@par Inputs:
*Two inputs, including:
* @li x1: A tensor. Must be one of the following types:
*     float16, float, double, bfloat16, int8, int16, int32, int64, qint8, qint16, qint32, qint64,
*     quint8, uin8, uint16, uint32, uint64, complex32, complex64, complex128, bool. \n

* @li x2: A tensor with the same dtype and shape as 'x1'. \n

*@par Attributes:
*@li rtol: An optional float.Defaults to 1e-05. \n
*@li atol: An optional float.Defaults to 1e-08. \n
*@li equal_nan: An optional bool.Defaults to false. \n

*@par Outputs:
*y: A tensor bool with the same shape as 'x1'. \n

*@par Third-party framework compatibility
*Compatible with the PyTorch operator isclose. \n
*/
REG_OP(IsClose)
    .INPUT(x1, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .INPUT(x2, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(rtol, Float, 1e-05f)
    .ATTR(atol, Float, 1e-08f)
    .ATTR(equal_nan, Bool, false)
    .OP_END_FACTORY_REG(IsClose)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_CLOSE_H_

