/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
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

/* !
 * \file sub_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SUB_V2_H_
#define OPS_OP_PROTO_INC_SUB_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns x1 - x2.
*@par Inputs:
*Two inputs, including:
* @li x1: A NCHW or NHWC Tensor. Must be one of the following types: float32.
* @li x2: A NCHW or NHWC Tensor. Must be one of the following types: float32. \n

*@par Outputs:
*y: A NCHW or NHWC Tensor. Must be one of the following types: float32.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SubV2.
*/
REG_OP(SubV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(SubV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SUB_V2_H_