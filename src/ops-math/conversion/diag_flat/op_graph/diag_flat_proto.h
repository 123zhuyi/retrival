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
 * \file diag_flat_proto.h
 * \brief
 */
#ifndef OPS_CONVERSION_DIAG_FLAT_GRAPH_PLUGIN_DIAG_FLAT_PROTO_H_
#define OPS_CONVERSION_DIAG_FLAT_GRAPH_PLUGIN_DIAG_FLAT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Create a diagonal tensor
* @par Inputs:
* One input, include:
* x: A Tensor. Must be one of the
*     following types:
*     float16, float32, double, int8, int16,int32, int64, complex32, complex64, complex128.

* @par Attributes:
* @li diagonal: An optional int value. Defaults to 0, this attribute controls which diagonal to consider:
*               If diagonal = 0, it is the main diagonal.
*               If diagonal > 0, it is above the main diagonal.
*               If diagonal < 0, it is below the main diagonal.

* @par Outputs:
* y: A mutable Tensor. Has the same type as "x".
* @see DiagFlat()
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Diag.
*/
REG_OP(DiagFlat)
    .INPUT(x, "T")
    .ATTR(diagonal, Int, 0)
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE,
                             DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                             DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                             DT_BOOL, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(DiagFlat)

} // namespace ge

#endif