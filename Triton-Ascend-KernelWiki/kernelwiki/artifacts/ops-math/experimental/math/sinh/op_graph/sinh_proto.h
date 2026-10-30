/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
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
 * \file sinh_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SINH_H_
#define OPS_OP_PROTO_INC_SINH_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief Returns element-wise hyperbolic sine of "x".
 *@par Inputs:
 *One input, including:
 * @li x: A ND Tensor. Must be one of the following types: float32,float16.\n
 *@par Outputs:
 *y: A ND Tensor. Has the same dtype as "x".\n
 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator Sinh.
 */
REG_OP(Sinh)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(Sinh)

} // namespace ge

#endif // OPS_OP_PROTO_INC_Sinh_H_