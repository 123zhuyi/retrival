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
 * \file reduce_std_with_mean.cpp
 * \brief
 */
#include "reduce_std_with_mean.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"


using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceStdWithMean);

const aclTensor* ReduceStdWithMean(
    const aclTensor* self, const aclTensor* mean, const aclIntArray* dim, int64_t correction, bool keepdim, bool invert,
    float eps, aclOpExecutor* executor)
{
    L0_DFX(ReduceStdWithMean, self, mean, dim, correction, keepdim, invert, eps);
    auto stdWithMeanOut =
        executor->AllocTensor(self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
    CHECK_RET(stdWithMeanOut != nullptr, nullptr);
    bool unbiased = true;
    if (correction == 0) {
        unbiased = false;
    }
    INFER_SHAPE(
        ReduceStdWithMean, OP_INPUT(self, mean), OP_OUTPUT(stdWithMeanOut),
        OP_ATTR(dim, unbiased, keepdim, invert, eps, correction));
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ReduceStdWithMean, OP_INPUT(self, mean), OP_OUTPUT(stdWithMeanOut),
        OP_ATTR(dim, unbiased, keepdim, invert, eps, correction));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceStdWithMeanAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return stdWithMeanOut;
}
} // namespace l0op
