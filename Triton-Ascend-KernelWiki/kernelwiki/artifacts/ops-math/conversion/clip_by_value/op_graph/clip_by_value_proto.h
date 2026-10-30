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
 * \file clip_by_value_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CLIP_BY_VALUE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CLIP_BY_VALUE_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Clips tensor values to a specified min and max.
* When the input is bfloat16, float16, float32, int32 or int64, broadcasting operations are supported.  \n

* @par Inputs:
* Three inputs, including:
* @li x: A ND tensor of type complex128, complex64, double, float32, float16, int16，
* int32, int64, int8, qint32, qint8, quint8, uint16, uint8, bfloat16, complex32.
* @li clip_value_min: A ND tensor of the same dtype as "x".
* @li clip_value_max: A ND tensor of the same dtype as "x". \n

* @par Outputs:
* y: A ND tensor. Has the same dtype as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ClipByValue.
*/
REG_OP(ClipByValue)
    .INPUT(
        x, TensorType(
               {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8,
                DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .INPUT(
        clip_value_min, TensorType(
                            {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                             DT_INT8, DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .INPUT(
        clip_value_max, TensorType(
                            {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                             DT_INT8, DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .OUTPUT(
        y, TensorType(
               {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8,
                DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(ClipByValue)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_CLIP_BY_VALUE_OPS_H_
