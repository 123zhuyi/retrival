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
 * \file minimum_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MINIMUM_H_
#define OPS_OP_PROTO_INC_MINIMUM_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns min(x1, x2.
*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: float32,float16,int32,int16,bfloat16,int8.
* @li x2: A ND Tensor. Must be one of the following types: float32,float16,int32,int16,bfloat16,int8. \n

*@par Outputs:
*y: A ND Tensor. Must be one of the following types: float32,float16,int32,int16,bfloat16,int8.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Minimum.
*/
REG_OP(Minimum)

    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16, DT_INT8}))
    .OP_END_FACTORY_REG(Minimum)

} // namespace ge

#endif // OPS_OP_PROTO_INC_Minimum_H_