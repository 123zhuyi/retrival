/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_COORDINATES_1D_TO_2D_DATA_TRANSFORMATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_COORDINATES_1D_TO_2D_DATA_TRANSFORMATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Converts 1D coordinate to 2D coordinate based on shape information.

* @par Inputs:
* Two inputs:
* @li x: 1D coordinate index value.
*   Must be one of the following types: int32, int64, uint64. \n
* @li shape: Shape information with 4 elements (N, D, H, W).
*   Must be one of the following types: int32, int64, uint64. \n

* @par Outputs:
* Three outputs:
* @li row: Row index result. \n
 * @li col: Column index result. \n
 * @li n: Column count (W dimension value). \n

* @attention Constraints:
* @li input[shape] element count must be 4.
* @li input[x] and input[shape] data type must be the same.
* @li shape[3] (W dimension) cannot be 0. \n

* @par Third-party framework compatibility
* Compatible with internal Huawei operator. \n
*/
REG_OP(Coordinates1DTo2D)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(row, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(col, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(n, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OP_END_FACTORY_REG(Coordinates1DTo2D)
}  // namespace ge

#endif