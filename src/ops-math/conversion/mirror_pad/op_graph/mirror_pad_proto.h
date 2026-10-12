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
 * \file mirror_pad_proto.h
 * \brief
 */

#ifndef MIRROR_PAD_PROTO_H_
#define MIRROR_PAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fills the tensor with the mirror value.
* @par Inputs:
* @li x: The tensor to be padded. Format support ND,
* Type must be one of the following: int8, uint8, int16, uint16, int32, uint32, int64, uint64,
* bfloat16, float16, float, double, bool, complex64, complex128, hifloat8, float8_e5m2, float8_e4m3fn, float8_e8m0.
* @li paddings: A two-column matrix specifying the padding sizes.
* Arranged as [[leftpad_0, rightpad_0], [leftpad_1, rightpad_1], ...]
* The number of rows has the same rank as "x", type must be int32 or int64.

* @par Attributes:
* @li mode: Either "REFLECT" or "SYMMETRIC". In reflect mode the padded regions
* do not include the borders, while in symmetric mode the padded regions
* do include the borders.

* @par Outputs:
* y: The padded tensor with the same type as "x".
* y.shape[i] = x.shape[i] + leftpad_i + rightpad_i, where y.shape[i] >= 0.

* @attention Constraints:
* "symmetric" mode: the leftpad_i and rightpad_i should be in [-x.shape[i], x.shape[i]]
* "reflect" mode: the leftpad_i and rightpad_i should be in [-x.shape[i], x.shape[i])

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MirrorPad.
*/

REG_OP(MirrorPad)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16,
                          DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, DT_COMPLEX64, DT_COMPLEX128, DT_HIFLOAT8,
                          DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .INPUT(paddings, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16,
                           DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, DT_COMPLEX64, DT_COMPLEX128, DT_HIFLOAT8,
                           DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPad)
} // namespace ge

#endif
