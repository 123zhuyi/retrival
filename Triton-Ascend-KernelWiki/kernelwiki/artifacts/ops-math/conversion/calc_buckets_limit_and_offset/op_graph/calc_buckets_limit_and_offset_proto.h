/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CALC_BUCKETS_LIMIT_AND_OFFSET_DATA_TRANSFORMATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CALC_BUCKETS_LIMIT_AND_OFFSET_DATA_TRANSFORMATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Calculate buckets limit and offset. \n

* @par Inputs:
* Three inputs, including:
* @li bucket_list: A 1-D tensor of type int32 with the value of ivf_counts and ivf_offset index. \n
* @li ivf_counts: A 1-D tensor of type int32 with the value of ivf counts. \n
* @li ivf_offset: A 1-D tensor of type int32 or int64 with the value of ivf offset. \n

* @par Attributes:
* total_limit: A int64 type maximum value of the sum of ivf_counts corresponding to bucket_list. \n

* @par Outputs:
* @li buckets_limit: A 1-D tensor of type int32 with the sum <= total_limit. \n
* @li buckets_offset: A 1-D tensor of type int32 or int64 with the value of ivf_offset corresponding to bucket_list. \n
*/
REG_OP(CalcBucketsLimitAndOffset)
    .INPUT(bucket_list, TensorType({DT_INT32}))
    .INPUT(ivf_counts, TensorType({DT_INT32}))
    .INPUT(ivf_offset, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(buckets_limit, TensorType({DT_INT32}))
    .OUTPUT(buckets_offset, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(total_limit, Int)
    .OP_END_FACTORY_REG(CalcBucketsLimitAndOffset)
} // namespace ge

#endif
