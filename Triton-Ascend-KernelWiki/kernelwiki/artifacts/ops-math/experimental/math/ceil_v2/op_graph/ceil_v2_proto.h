/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file ceil_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_DIV_H_
#define OPS_OP_PROTO_INC_DIV_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Returns element-wise smallest integer not less than "x".

*@par Inputs:

x: A ND Tensor of type float16 or float32 . \n
*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Ceil.
*/

REG_OP(CeilV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(CeilV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CeilV2_H_