/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPLIT_V_COMBINATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPLIT_V_COMBINATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Splits a tensor along dimension "split_dim" into "num_split"
  smaller tensors according to "size_splits" .

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor.
* Must be one of the types:float16, float32, double, int64, int32, uint8,
  uint16, uint32, uint64, int8, int16, bool, complex64, complex128, qint8,
  quint8, qint16, quint16, qint32, string, bfloat16.
* @li size_splits: Must be one of the IndexNumberType:int32, int64.
  Specifies a list containing the sizes of each output tensor along the split dimension.
  The elements in "size_splits" sum to the size of dimension "split_dim".
* @li split_dim: Must be the following type:int32, int64. Specifies the
  dimension along which to split. Must be in the range [-len(x.shape), len(x.shape)) . \n

* @par Attributes:
* @li num_split: A required int includes all types of int. Specifies the number of output tensors.
  No default value . \n

* @par Outputs:
* @li y:  Dynamic output.A list of output tensors.
  Has the same type and format as "x" . \n

* @attention Constraints:
* @li Each element in "size_splits" is greater than or equal to 1.
* @li The length of "size_splits" is equal to the value of "num_split".
* @li The elements in "size_splits" sum to the size of dimension "split_dim" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SplitV.
*/
REG_OP(SplitV)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                          DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                          DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                          DT_BF16, DT_BOOL, DT_STRING}))
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                                   DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                                   DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                                   DT_BF16, DT_BOOL, DT_STRING}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)
}  // namespace ge

#endif