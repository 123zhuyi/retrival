/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_BATCH_TO_SPACE_N_D_PROTO_H_
#define OP_PROTO_BATCH_TO_SPACE_N_D_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Permutes data from batch into blocks of spatial data and then prunes them.
* The values from the batch dimension are moved in spatial blocks to the height and width dimensions.
* And then prunes the height and width dimensions.

* @par Inputs:
* @li x: A ND tensor, must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
* @li block_shape: A 1D tensor with shape [M], support int32 or int64.
* @li crops: A 2D tensor with shape [M, 2], support int32 or int64. \n

* @par Outputs:
* y: A ND tensor, the same type as "x". \n

* @attention Constraints:
* If N is 4 and M is 2: \n
* The size of the first dimension of input "x" must be divisible by the product of all elements in block_shape. \n
* "y" is a 4D shape [batch, height, width, depth], batch = x.shape[0] / (block_shape[0] * block_shape[1]),
* depth = x.shape[3], height = height_pad - crop_top - crop_bottom, width = width_pad - crop_left - crop_right
* where height_pad = x.shape[1] * block_shape[0], width_pad = x.shape[2] * block_shape[1],
* crop_top = crops[0][0], crop_bottom = crops[0][1], crop_left = crops[1][0], crop_right = crops[1][1]
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchToSpaceND.
*/
REG_OP(BatchToSpaceND)
    .INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL}))
    .OP_END_FACTORY_REG(BatchToSpaceND)
} // namespace ge

#endif
