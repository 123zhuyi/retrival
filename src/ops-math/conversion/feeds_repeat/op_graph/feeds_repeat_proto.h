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
 * \file feeds_repeat_proto.h
 * \brief
 */
#ifndef OPS_CONVERSION_FEEDS_REPEAT_GRAPH_PLUGIN_FEEDS_REPEAT_PROTO_H_
#define OPS_CONVERSION_FEEDS_REPEAT_GRAPH_PLUGIN_FEEDS_REPEAT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Repeat every row of input 'feeds' different times with padding in the end.
* @par Inputs:
* Two inputs:
* @li feeds: A Tensor with ND format. Only support float, float16, bfloat16 for now. Other types include int8, int16,
int32, int64, uint8, uint16, uint32, uint64, bool will be supported later.
* @li feeds_repeat_times: A Tensor with ND format. Support int32, int64. Describe repeat times for every row of input
'feeds'.

* @par Attributes:
* output_feeds_size: An required int, specifying the dim0 of y, describing padding space.

* @par Outputs:
* y: A Tensor, which is the same dtype as feeds. Dim0 is output_feeds_size, other dims are same as feeds.

* @attention Constraints:
* @li The length of feeds_repeat_times must be same as feeds' dim0 size.
* @li The byte of feeds_repeat_times cannot reach about 64KB.
*/
REG_OP(FeedsRepeat)
    .INPUT(feeds, TensorType::BasicType())
    .INPUT(feeds_repeat_times, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(output_feeds_size, Int)
    .OP_END_FACTORY_REG(FeedsRepeat)

} // namespace ge

#endif