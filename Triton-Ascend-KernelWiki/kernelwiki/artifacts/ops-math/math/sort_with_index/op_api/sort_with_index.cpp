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
 * \file sort_with_index.cpp
 * \brief
 */
#include "sort_with_index.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SortWithIndex);

static const int64_t AXIS_LIMIT = 8;

const std::tuple<aclTensor*, aclTensor*> SortWithIndex(const aclTensor* self, const aclTensor* index, int64_t axis, bool descending, bool stable,
    aclOpExecutor* executor)
{
    auto selfShape = self->GetViewShape();
    auto selfFormat = self->GetViewFormat();

    int64_t dimSize = static_cast<int64_t>(selfShape.GetDimNum());
    if (dimSize > AXIS_LIMIT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor self dimension size must be in range [0, 8]. Current size is [%ld].",
            dimSize);
    }

    auto values = executor->AllocTensor(selfShape, self->GetDataType(), selfFormat);
    auto sorted_index = executor->AllocTensor(selfShape, DataType::DT_INT32, selfFormat);

    L0_DFX(SortWithIndex, self, index, axis, descending, stable, values, sorted_index);
    if ((dimSize!= axis + 1) && (axis != -1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "axis must equal to the (number of dimensions - 1 ) or -1.");
    }
    ADD_TO_LAUNCHER_LIST_AICORE(SortWithIndex, OP_INPUT(self, index), OP_OUTPUT(values, sorted_index), OP_ATTR(axis, descending, stable));
    
    return std::tie(values, sorted_index);
}
}
