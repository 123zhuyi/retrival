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
 * \file stateless_sample_multinomial.cpp
 * \brief Op API implementation for StatelessSampleMultinomial
 */

#include "stateless_sample_multinomial.h"
#include "aclnn_kernels/cast.h"
#include "math/add/op_api/add.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(StatelessSampleMultinomial);
OP_TYPE_REGISTER(SimThreadExponential);

const aclTensor* StatelessSampleMultinomial(
    const aclTensor* xTensor,
    const aclTensor* seedTensor,
    const aclTensor* offsetTensor,
    int64_t numsamples,
    aclOpExecutor* executor)
{
    L0_DFX(StatelessSampleMultinomial, xTensor, seedTensor, offsetTensor);

    auto outShape = xTensor->GetViewShape();
    auto dimNum = outShape.GetDimNum();
    CHECK_RET(dimNum == 1 || dimNum == 2, nullptr);
    outShape.SetDim(dimNum - 1, numsamples);
    aclTensor* out = executor->AllocTensor(outShape, DataType::DT_INT64, xTensor->GetViewFormat());
    CHECK_RET(out != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessSampleMultinomial,
        OP_ATTR_NAMES({"num_samples"}),
        OP_INPUT(xTensor, seedTensor, offsetTensor),
        OP_OUTPUT(out),
        OP_ATTR(numsamples));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return out;
}

const aclTensor* Run950AicoreExponentialWithoutReplacement(
    const aclTensor* self, int64_t seed, int64_t offset, float lambd, aclOpExecutor* executor)
{
    aclTensor* out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    CHECK_RET(out != nullptr, nullptr);

    auto shape = self->GetViewShape();
    int64_t count = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        count *= shape.GetDim(i);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SimThreadExponential,
        OP_INPUT(out),
        OP_OUTPUT(out),
        OP_ATTR(count, lambd, seed, offset));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return out;
}

} // namespace l0op
