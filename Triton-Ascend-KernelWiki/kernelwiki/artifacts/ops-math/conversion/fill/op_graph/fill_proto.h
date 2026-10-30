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
 * \file fill_proto.h
 * \brief
 */
#ifndef OP_PROTO_FILL_H_
#define OP_PROTO_FILL_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
* @brief Creates a tensor filled with a scalar value.
* This operation creates a tensor of shape "dims" and fills it with "value".
*
* @par Inputs:
* @li dims: A 1D tensor of types int32 or int64. Represents the shape of the output tensor .
        The size of each dimension must be less than or equal to 8. \n

* @li value: A 0D scalar. Specifies the value to fill the returned tensor.
*    Must be one of the following types:
*    bfloat16, float16, float32, double, int32, uint8, int16, int8, complex64, int64, bool,
*    qint8, quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, string.
*
* @par Outputs:
* y: A tensor. Has the same type as "value".
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Fill.
* @li Compatible with the Caffe operator Filler.
*
*/
REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))
    .OP_END_FACTORY_REG(Fill)

} // namespace ge
#endif // OP_PROTO_FILL_H_