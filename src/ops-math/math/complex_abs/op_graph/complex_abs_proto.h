/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file complex_abs_proto.h
 * \brief
 */
#ifndef OP_PROTO_COMPLEX_ABS_PROTO_H_
#define OP_PROTO_COMPLEX_ABS_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes the complex absolute value of a tensor.
 *
 * @par Inputs:
 * x: A tensor of complex numbers. A Tensor of type complex32, complex64, or complex128.
 *
 * @par Attributes:
 * Tout: A Type attr, representing the type of output, do not use. Default is DT_FLOAT.
 *
 * @par Outputs:
 * y: A tensor that is the absolute value of each element in x.
 * A Tensor of type float16 (when x is complex32), float32 (when x is complex64),
 * or double (when x is complex128).
 *
 * @attention Constraints:
 * @li The shape of output "y" must be the same as the shape of input "x".
 * @li Empty tensor (shape contains 0) is not supported.
 *
 * @par Third-party framework compatibility
 * Compatible with TensorFlow ComplexAbs operator.
 */
#ifndef OPS_PROTO_DEF_COMPLEXABS
#define OPS_PROTO_DEF_COMPLEXABS
REG_OP(ComplexAbs)
    .INPUT(x, TensorType({DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(Tout, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(ComplexAbs)
#endif // OPS_PROTO_DEF_COMPLEXABS
} // namespace ge
#endif // OP_PROTO_COMPLEX_ABS_PROTO_H_
