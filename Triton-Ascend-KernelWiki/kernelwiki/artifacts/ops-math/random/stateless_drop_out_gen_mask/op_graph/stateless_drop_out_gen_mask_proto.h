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
 * \file stateless_drop_out_gen_mask_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_DROP_OUT_GEN_MASK_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_DROP_OUT_GEN_MASK_PROTO_H_

#include <vector>

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Generate stateless random bit mask for dropout . \n

* @par Inputs:
include:
* @li shape:The shape of the output tensor.
* @li prob:0-D. Number of bit 1 . \n
* @li seed:Frist seed to avoid seed collision.
* @li seed1:Second seed to avoid seed collision . \n
* @li offset:Initial offset of random number . \n

* @par Outputs:
*y:Output (1-D) random number using uint data format . \n

* @attention Constraints:
*The output is aligned with 128 bits.

* @see StatelessDropOutGenMask()
*/
REG_OP(StatelessDropOutGenMask)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(prob, TensorType({ DT_FLOAT16, DT_FLOAT, DT_BF16 }))
    .INPUT(seed, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(seed1, TensorType({ DT_INT32, DT_INT64 }))
    .OPTIONAL_INPUT(offset, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_UINT8 }))
    .OP_END_FACTORY_REG(StatelessDropOutGenMask)
}   // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_
