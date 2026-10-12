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
 * \file equal_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_EQUAL_H_
#define OPS_OP_PROTO_INC_EQUAL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns the truth value of (x = y) element-wise. Support broadcasting operations. \n

*@par Inputs:
* Two inputs, including:
*@li x1: A ND Tensor. Must be one of the following types:
*    bfloat16, float16, float32, int32, int8, uint8, double, int16, int64, complex64,
*    complex128, quint8, qint8, qint32, string, bool. the format can be [NCHW, NHWC, ND]
*@li x2: A ND Tensor of the same dtype and format as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the bool dtype. True means x1 == x2, false means x1 != x2.

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Equal.
*/
REG_OP(Equal)
    .INPUT(
        x1, TensorType(
                {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8, DT_DOUBLE, DT_INT16, DT_INT64,
                 DT_COMPLEX64, DT_COMPLEX128, DT_QUINT8, DT_QINT8, DT_QINT32, DT_STRING, DT_BOOL}))
    .INPUT(
        x2, TensorType(
                {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8, DT_DOUBLE, DT_INT16, DT_INT64,
                 DT_COMPLEX64, DT_COMPLEX128, DT_QUINT8, DT_QINT8, DT_QINT32, DT_STRING, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Equal)

} // namespace ge

#endif // OPS_OP_PROTO_INC_EQUAL_H_
