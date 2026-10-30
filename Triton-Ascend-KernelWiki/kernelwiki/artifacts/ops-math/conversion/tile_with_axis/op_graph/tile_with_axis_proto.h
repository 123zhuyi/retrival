/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_TILE_WITH_AXIS_H_
#define OPS_OP_PROTO_TILE_WITH_AXIS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Extends the input with copies of data along a specified dimension. For example: \n
* (1) If x = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], with shape (2, 3, 2); \n
* (2) axis = 1; \n
* (3) tiles = 2; \n
* (4) Then, y = [[[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]], [[7, 8],
* [9, 10], [11, 12], [7, 8], [9, 10], [11, 12]]],
* with shape (2, 6, 2).

* @par Inputs:
* One input:
* x: A Tensor with any format. Must be one of the following types:
* bfloat16, float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64 . \n

* @par Attributes:
* @li axis: An optional int, specifying the axis to tile. Defaults to 1.
* @li tiles: A required int, specifying the number of copies (tiles) to output . \n

* @par Outputs:
* y: A Tensor with the same type and format of x. \n

* @attention Constraints:
* @li "axis" must be within the rank of the input tensor.
* @li "tiles" must be greater than 1.
* @par Third-party framework compatibility
* Compatible with the Caffe operator Tile.
*/
REG_OP(TileWithAxis)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT64, DT_UINT32,
                          DT_UINT16, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT64, DT_UINT32,
                           DT_UINT16, DT_UINT8}))
    .ATTR(axis, Int, 1)
    .REQUIRED_ATTR(tiles, Int)
    .OP_END_FACTORY_REG(TileWithAxis)
} // namespace ge

#endif // OPS_OP_PROTO_TILE_WITH_AXIS_H_
