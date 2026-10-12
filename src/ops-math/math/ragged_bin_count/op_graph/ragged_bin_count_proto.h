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
 * \file ragged_bin_count_proto.h
 * \brief Proto definition for the RaggedBinCount operator.
 */
#ifndef OPS_OP_PROTO_INC_RAGGED_BIN_COUNT_H_
#define OPS_OP_PROTO_INC_RAGGED_BIN_COUNT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Counts occurrences, or accumulated weights, for every bin in each ragged row.
 *
 * @par Inputs:
 * @li splits: A 1D int64 tensor. Adjacent entries delimit each ragged row in values.
 * @li values: An int32 or int64 tensor containing bin indices.
 * @li size: A non-negative 1D tensor with shape [1] and the same dtype as values. Its only element specifies
 *     the number of bins.
 * @li weights: A tensor of rank at most 2, interpreted in flattened ND order. Zero elements mean unit weights;
 *     otherwise its element count must equal values. Rank and individual dimensions need not match values.
 *
 * @par Attributes:
 * @li binary_output: If true, every bin that occurs at least once is written as one. Defaults to false.
 *
 * @par Outputs:
 * @li output: A float32 tensor with shape [numel(splits) - 1, size[0]].
 *
 * @attention The public REG_OP retains the historical broad weights/output dtype declarations for IR compatibility.
 *     The supported native implementations accept float32 weights and produce float32 output.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow RaggedBincount operator.
 */
// The guard name deliberately keeps the snake-case spelling: it has to be byte-identical to the one
// canndev's ops/built-in/op_proto/inc/ops_proto_legacy.h uses for the same REG_OP, or the two
// definitions land in the same translation unit and OP_END_FACTORY_REG registers RaggedBinCount
// twice. canndev spells it OPS_PROTO_DEF_RAGGED_BIN_COUNT (one of only two of its 916 guards that
// keeps the underscores), so the repository's usual OPS_PROTO_DEF_<OPNAME> form would not pair.
#ifndef OPS_PROTO_DEF_RAGGED_BIN_COUNT
#define OPS_PROTO_DEF_RAGGED_BIN_COUNT
REG_OP(RaggedBinCount)
    .INPUT(splits, TensorType(DT_INT64))
    .INPUT(values, TensorType({DT_INT32, DT_INT64}))
    .INPUT(size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(weights, TensorType(DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE))
    .OUTPUT(output, TensorType(DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE))
    .ATTR(binary_output, Bool, false)
    .OP_END_FACTORY_REG(RaggedBinCount)
#endif // OPS_PROTO_DEF_RAGGED_BIN_COUNT

} // namespace ge

#endif // OPS_OP_PROTO_INC_RAGGED_BIN_COUNT_H_
