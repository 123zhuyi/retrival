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
 * \file add_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ADDV2_H_
#define OPS_OP_PROTO_INC_ADDV2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief Returns x1 + x2.
 *@par Inputs:
 *Two inputs, including:
 * @li x1: A tensor. Must be one of the following types: bfloat16, float16, float32, float64,
 *     uint8, int8, int16, int32, int64, complex64, complex128.
 * @li x2: A tensor of the same dtype as "x1".
 *
 *@attention Constraints:
 * AddV2 supports broadcasting.
 *
 *@par Outputs:
 * y: A tensor. Has the same dtype as "x1".
 *
 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator AddV2.
 *
 */
#ifndef OPS_PROTO_DEF_ADDV2
#define OPS_PROTO_DEF_ADDV2
REG_OP(AddV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX64, DT_BF16, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX64, DT_BF16, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX64, DT_BF16, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(AddV2)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_ADDV2_H_
