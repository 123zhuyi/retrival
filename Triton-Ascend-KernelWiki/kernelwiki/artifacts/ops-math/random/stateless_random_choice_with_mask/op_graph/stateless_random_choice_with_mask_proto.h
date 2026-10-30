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
 * \file stateless_random_choice_with_mask_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_STATELESS_RANDOM_CHOICE_WITH_MASK_OPS_H_
#define OPS_OP_PROTO_INC_STATELESS_RANDOM_CHOICE_WITH_MASK_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Generate stateless random choice for tensor input . \n

* @par Inputs:
include:
* @li x: A ND Tensor of type bool. Support 1D~5D. If x is true, this index will be selected.
* @li count: A ND Tensor of type int32. Support 1D. Must be greater than or equal to 0.
* @li seed: A ND Tensor of type int64. Support 1D, should be const data.
* @li offset: A ND Tensor of type int64. Support 1D, should be const data.

* @par Outputs:
* @li y: A ND Tensor of int32.
* @li mask: A ND Tensor of bool. If x is true, y is a valid index, otherwise it is 0.
*/

REG_OP(StatelessRandomChoiceWithMask)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(count, TensorType({DT_INT32}))
    .INPUT(seed, TensorType({DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(StatelessRandomChoiceWithMask)
}  // namespace ge

#endif  // OPS_OP_PROTO_INC_STATELESS_RANDOM_CHOICE_WITH_MASK_OPS_H_
