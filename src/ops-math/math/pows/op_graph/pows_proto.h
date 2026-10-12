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
 * \file pows_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_POWS_OPS_H_
#define OPS_OP_PROTO_INC_POWS_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Apply power operation for a scalar
* in manner of element-wise
* @par Inputs:
* Two inputs:
* @li x1: A tensor to be power
* @li x2: Another tensor contains a power scalar
* @par Outputs:
* @li y: A tensor which value are power with the scalar

* @par Third-party framework compatibility:
* New operator Pows.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Pows)
    .INPUT(x1, "T")
    .INPUT(x2, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Pows)

} // namespace ge

#endif