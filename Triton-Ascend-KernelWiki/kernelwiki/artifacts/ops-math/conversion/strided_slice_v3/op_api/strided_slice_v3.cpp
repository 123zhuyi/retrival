/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "strided_slice_v3.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

namespace l0op {
OP_TYPE_REGISTER(StridedSliceV3);

// AICORE算子kernel
const aclTensor* StridedSliceV3AiCore(
    const aclTensor* self, const aclTensor* begin, const aclTensor* end, const aclTensor* axis,
    const aclTensor* strideds, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(StridedSliceV3AiCore, self, begin, end, axis, strideds, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(StridedSliceV3, OP_INPUT(self, begin, end, axis, strideds), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "StridedSliceV3AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

const aclTensor* StridedSliceV3(
    const aclTensor* self, int64_t begin, int64_t end, int64_t axis, int64_t strideds, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
    auto beginTensor = executor->ConvertToTensor(&begin, 1, op::DataType::DT_INT64);
    auto endTensor = executor->ConvertToTensor(&end, 1, op::DataType::DT_INT64);
    auto axisTensor = executor->ConvertToTensor(&axis, 1, op::DataType::DT_INT64);
    auto stridesTensor = executor->ConvertToTensor(&strideds, 1, op::DataType::DT_INT64);

    INFER_SHAPE(StridedSliceV3, OP_INPUT(self, beginTensor, endTensor, axisTensor, stridesTensor), OP_OUTPUT(out));
    return StridedSliceV3AiCore(self, beginTensor, endTensor, axisTensor, stridesTensor, out, executor);
}

const aclTensor* StridedSliceV3V2(
    const aclTensor* self, const aclIntArray* begins, const aclIntArray* ends, const aclIntArray* axis,
    const aclIntArray* strideds, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
    auto beginTensor = executor->ConvertToTensor(begins, op::DataType::DT_INT64);
    auto endTensor = executor->ConvertToTensor(ends, op::DataType::DT_INT64);
    auto axisTensor = executor->ConvertToTensor(axis, op::DataType::DT_INT64);
    auto stridesTensor = executor->ConvertToTensor(strideds, op::DataType::DT_INT64);

    INFER_SHAPE(StridedSliceV3, OP_INPUT(self, beginTensor, endTensor, axisTensor, stridesTensor), OP_OUTPUT(out));
    return StridedSliceV3AiCore(self, beginTensor, endTensor, axisTensor, stridesTensor, out, executor);
}
} // namespace l0op
