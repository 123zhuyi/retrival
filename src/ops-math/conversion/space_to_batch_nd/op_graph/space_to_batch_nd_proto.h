/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPACE_TO_BATCH_ND_PROTO_H
#define SPACE_TO_BATCH_ND_PROTO_H

#include "graph/operator_reg.h"

/**
* @brief Zeros-pads and then permutes blocks of spatial data into batch.
* The values from the height and width dimensions are moved in spatial blocks to the batch dimension.
* After zeros-pads the height and width dimensions. \n

* @par Inputs:
* @li x: A ND tensor. Format is ND. Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
* @li block_shape: A 1D tensor with shape [M]. Format is ND. Support int32 or int64.
* @li paddings: A 2D tensor with shape [M, 2]. Format is ND. Support int32 or int64. \n

* @par Outputs:
* y: A tensor, the same type and format as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToBatchND.
*/
REG_OP(SpaceToBatchND)
    .INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(SpaceToBatchND)

#endif
