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
 * \file non_finite_check_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_NON_FINITE_CHECK_OPS_H_
#define OPS_OP_PROTO_INC_NON_FINITE_CHECK_OPS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Check if there are non-finite numbers (+inf/-inf/nan) in the tensor_list.
 * If there are, set found_flag to 1, otherwise, set found_flag to 0.
 * @par Inputs:
 * One input:
 * tensor_list: Dynamic input, A tensor list containing multiple ND format tensors,
 * Support 1D ~ 8D, dtype can be float16, bfloat16, float32.
 * The dtype of each tensor in the tensor_list must be consistent,
 * and the tensor_list can contain a maximum of 256 tensors.
 * @par Outputs:
 * found_flag: A tensor with only one element, the shape must be (1,), must be float.
 */
REG_OP(NonFiniteCheck)
    .DYNAMIC_INPUT(tensor_list, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(found_flag, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NonFiniteCheck)

} // namespace ge

#endif