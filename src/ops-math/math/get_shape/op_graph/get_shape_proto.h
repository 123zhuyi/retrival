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
 * \file get_shape_proto.h
 * \brief
 */

#ifndef GET_SHAPE_PROTO_H_
#define GET_SHAPE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief Returns the shape of one or more tensors. \n

 *@par Inputs:
 *x: A list of tensors. Must be one of the following types: float32、float16、int8、
 int16、uint16、uint8、int32、int64、uint32、uint64、bool、double. \n

 *@par Outputs:
 *y: A tensor. The shape of the input tensors. Output type is int32. \n

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator GetShape.
 */

#ifndef OPS_PROTO_DEF_GETSHAPE
#define OPS_PROTO_DEF_GETSHAPE
REG_OP(GetShape)
    .DYNAMIC_INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                                  DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(GetShape);

#endif
} // namespace ge

#endif // GET_SHAPE_PROTO_H_
