/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_SPACE_TO_BATCH_PROTO_H_
#define OP_PROTO_SPACE_TO_BATCH_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief SpaceToBatch divides spatial data into blocks and moves them to the batch dimension.
 * This is the inverse of BatchToSpaceND.
 *
 * @par Inputs:
 * @li x: A 4D NHWC tensor [N, H_in, W_in, C], types: float16, float32, double, int64, int32,
 * int8, uint8, int16, uint16.
 * @li paddings: A 2D int tensor with shape [2, 2] = [[pad_top, pad_bottom], [pad_left, pad_right]].
 *
 * @par Attributes:
 * @li block_size: A required int. The spatial block size.
 *
 * @par Outputs:
 * y: A 4D NHWC tensor [N*block_size*block_size, H_out, W_out, C] where
 * H_out = (H_in + pad_top + pad_bottom) / block_size,
 * W_out = (W_in + pad_left + pad_right) / block_size.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator space_to_batch.
 */
REG_OP(SpaceToBatch)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(SpaceToBatch)
} // namespace ge

#endif
