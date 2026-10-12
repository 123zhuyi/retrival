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
 * \file ger_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_GERV2_H_
#define OPS_OP_PROTO_INC_GERV2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(GerV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16}))
    .INPUT(A, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16}))
    .OUTPUT(z, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16}))
    .ATTR(alpha, Float, 1.0)
    .OP_END_FACTORY_REG(GerV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_GERV2_H_