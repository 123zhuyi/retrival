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
 * \file diag_v2_proto.h
 * \brief
 */
#ifndef OP_PROTO_DIAG_V2_H_
#define OP_PROTO_DIAG_V2_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Create a diagonal tensor

* @par Inputs:
* One input, include:
* x: A mutable Tensor must be 2D tensor. Type must be one of the
*     following types:
*     DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_UINT16, DT_UINT32, DT_UINT64,
      DT_INT8, DT_UINT8, DT_DOUBLE, DT_BOOL,
      DT_COMPLEX32, DT_COMPLEX128, DT_COMPLEX64 . \n

* @par Attributes:
* diagonal: A optional int32. Specifies the position of output tensors'value. Defaults to "0" . \n

* @par Outputs:
* y: A mutable Tensor. Has the same type as "x" . \n

* @see Diag()
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Diag.
*/
REG_OP(DiagV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                          DT_FLOAT, DT_FLOAT16, DT_BF16, DT_DOUBLE, DT_BOOL,
                          DT_COMPLEX32, DT_COMPLEX128, DT_COMPLEX64}))
    .ATTR(diagonal, Int, 0)
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                           DT_FLOAT, DT_FLOAT16, DT_BF16, DT_DOUBLE, DT_BOOL,
                           DT_COMPLEX32, DT_COMPLEX128, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(DiagV2)

}  // namespace ge


#endif  // OP_PROTO_DIAG_V2_H_