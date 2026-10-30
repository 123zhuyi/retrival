/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Mindspore project.
 * Copyright 2019-2020 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef RIGHT_SHIFT_PROTO_H
#define RIGHT_SHIFT_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Element-wise computes the bitwise right-shift of x and y . \n

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
are 0D scalars.
* @li x: A Tensor. Must be one of the following types: int8, int16, int32,
int64, uint8, uint16, uint32, uint64.
* @li y: A Tensor. Has the same type as "x".  \n

*@par Outputs:
* z: A Tensor. Has the same type as "x".  \n

*@attention Constraints:
*Unique runs on the Ascend AI CPU, which delivers poor performance.  \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator RightShift.
*/

REG_OP(RightShift)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(z, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(RightShift)
} // namespace ge

#endif