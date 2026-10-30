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
 * \file not_equal_proto.h
 * \brief
 */

#ifndef NOT_EQUAL_H_
#define NOT_EQUAL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns the truth value of (x1 != x2) element-wise. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor with TensorType::RealNumberType().
* @li x2: A ND Tensor to be compared to "x1", and the data type is the same as "x1".

*@par Outputs:
*y: A ND Tensor. Has the bool dtype. True means x1 != x2, false means x1 == x2.

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator NotEqual.
*/
REG_OP(NotEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(NotEqual)
} // namespace ge

#endif