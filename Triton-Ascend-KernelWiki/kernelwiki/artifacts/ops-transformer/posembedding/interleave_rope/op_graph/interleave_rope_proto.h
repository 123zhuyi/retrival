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
 * \file interleave_rope_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_INTERLEAVEROPE_OPS_H_
#define OPS_OP_PROTO_INC_INTERLEAVEROPE_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
 * @brief The fusion operator of Interleave RotaryPositionEmbedding.
 *
 * @par Inputs:
 * @li x: A tensor. The type support float16, bfloat16. Format: ND.
 * @li cos: A tensor. The type support float16, bfloat16. Format: ND.
 * @li sin: A tensor. The type support float16, bfloat16. Format: ND.
 *
 * @par Outputs:
 * y: A tensor. The type support float16, bfloat16. Format: ND.
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */

REG_OP(InterleaveRope)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(InterleaveRope)

} // namespace ge

#endif
