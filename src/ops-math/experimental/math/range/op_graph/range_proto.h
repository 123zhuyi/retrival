/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Zhi <@hitLeechi>
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
 * \file range.cpp
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_RANGE_OPS_H_
#define OPS_OP_PROTO_INC_RANGE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Generates values in an interval .

*@par Inputs:
* Three ND inputs, including:
*@li start: A Scalar, for the first entry in the range.
* Type must be one of the following types:
* float32.
*@li end: A Scalar, for the last entry in the range.
* Type must be one of the following types:
* float32.
*@li step: A Scalar, for the last entry in the range. \n
* float32.
*@par Outputs:
*z: A 1D Tensor, Type is float32. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator RANGE.
*/
REG_OP(Range)
    .INPUT(start, TensorType({ge::DT_FLOAT}))
    .INPUT(end, TensorType({ge::DT_FLOAT}))
    .INPUT(step, TensorType({ge::DT_FLOAT}))
    .OUTPUT(z, TensorType({ge::DT_FLOAT}))
    .OP_END_FACTORY_REG(Range)

} // namespace ge

#endif // OPS_OP_PROTO_INC_RANGE_OPS_H_