/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
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
 * \file fill_v2_proto.h
 * \brief
 */
#ifndef OPS_CONVERSION_FILL_V2_GRAPH_PLUGIN_FILL_V2_PROTO_H_
#define OPS_CONVERSION_FILL_V2_GRAPH_PLUGIN_FILL_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fill all elements of a tensor with a scalar value inplace. \n

* @par Inputs:
* @li x: A tensor. Must be one of the following types:float16, float32,
                                                int16, int32, bfloat16. \n
* @li fill_value: A tensor. Scalar value to be filled, Must have the same type as "x". \n

* @par Outputs:
* x: A tensor. Refers to the same tensor as input "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Fill.
*/
REG_OP(FillV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT32}))
    .INPUT(fill_value, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT32}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT32}))
    .OP_END_FACTORY_REG(FillV2)

} // namespace ge

#endif