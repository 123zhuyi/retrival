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
 * \file batch_to_space_proto.h
 * \brief
 */
#ifndef BATCH_TO_SPACE_PROTO_H_
#define BATCH_TO_SPACE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief BatchToSpace rearranges data from the batch dimension into spatial blocks.
 *
 * @par Inputs:
 * Two inputs, including:
 * @li x: A 4D tensor, Format support ND, Must be one of the following types:
 * float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
 * int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
 * @li crops: A 2D Tensor of shape [2, 2] = [[crop_top, crop_bottom], [crop_left, crop_right]].
 *     Values are non-negative integers. Support int32 or int64.
 *
 * @par Attributes:
 * @li block_size: An int >= 1, specifying the size of the spatial block.
 *
 * @par Outputs:
 * y: A 4D Tensor of shape [N, H_out, W_out, C] where
 *    H_out = H_in * block_size - crop_top - crop_bottom
 *    W_out = W_in * block_size - crop_left - crop_right
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator BatchToSpace.
 */
REG_OP(BatchToSpace)
    .INPUT(x, TensorType::BasicType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(BatchToSpace)
} // namespace ge

#endif // BATCH_TO_SPACE_PROTO_H_
