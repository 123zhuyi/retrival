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
 * \file trilu_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_TRILU_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_TRILU_OPS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Returns the lower or upper triangular part of a matrix (2-D tensor) or batch of matrices input \n

*@par Inputs:
* @li x: A tensor, which supports 2-8 dimensions or be empty. Must be one of the following types:
* float16, bfloat16, float32, double, int32, uint8, int16, int8, int64,
* qint8, quint8, qint32, quint16, qint16, uint16, uint32, uint64, bool. \n

* @li k: An optional input tensor indicates the diagonal to consider. Must be int32 or int64 type. If no input,
* will be considerer as 0.

* @par Attributes:
* upper: An attribute indicates which part of the triangular matrix to be returned. Must be int32 type.
* If upper is 1, returns the upper triangular matrix,
* else if upper is 0, returns the lower triangular matrix. Default is 0. \n

* @par Outputs:
* y: A tensor. Has the same type and shape as "x" . \n

*/
#ifndef OPS_PROTO_DEF_TRILU
#define OPS_PROTO_DEF_TRILU
REG_OP(Trilu)
    .INPUT(x, "T")
    .OPTIONAL_INPUT(k, "T_K")
    .ATTR(upper, Int, 0)
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16,   DT_FLOAT,  DT_DOUBLE, DT_INT32,     DT_UINT8,    DT_INT16,
                             DT_INT8,    DT_INT64,  DT_QINT8,  DT_QUINT8, DT_QINT32,    DT_QUINT16,  DT_QINT16,
                             DT_UINT16,  DT_UINT32, DT_UINT64, DT_BOOL,   DT_COMPLEX32, DT_COMPLEX64}))
    .DATATYPE(T_K, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Trilu)
#endif
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_TRILU_OPS_H_
