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
 * \file dynamic_partition_proto.h
 * \brief
 */
#ifndef OPS_OP_GRAPH_INC_DYNAMIC_PARTITION_OPS_H_
#define OPS_OP_GRAPH_INC_DYNAMIC_PARTITION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Partitions "x" into "num_partitions" tensors using indices from "partitions".

*@par Inputs:
*Including:
* @li x: The tensor to be sliced. Must be one of the following types:
*DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
*DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE,
*DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN.
* @li partitions: A tensor of type DT_INT32, with any shape.

*@par Attributes:
*num_partitions: An optional attribute of type int, specifying the count of output, which must meet >= 1. Defaults to "1".

*@par Outputs:
*y: A list of tensors with same data type of x.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator DynamicPartition.
*/
REG_OP(DynamicPartition)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE,\
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(partitions, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE,\
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .ATTR(num_partitions, Int, 1)
    .OP_END_FACTORY_REG(DynamicPartition)

} // namespace ge

#endif