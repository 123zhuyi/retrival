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
 * \file sort_with_index.cpp
 * \brief SortWithIndex L0 (internal) op-api implementation (experimental ascend910b native).
 *
 * L0 flow: shape derivation -> AllocTensor (y follows self dtype, sorted_index follows index dtype)
 *          -> ADD_TO_LAUNCHER_LIST_AICORE(SortWithIndex, ...). Output shapes equal the input shapes
 *          (in-place permutation along the last axis); see op_host/sort_with_index_infershape.cpp.
 */

#include "sort_with_index_l0.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SortWithIndex);

const std::tuple<aclTensor*, aclTensor*> SortWithIndex(
    const aclTensor* self, const aclTensor* index, const int64_t axis, const bool descending, const bool stable, aclOpExecutor* executor)
{
    auto selfShape = self->GetViewShape();
    auto selfFormat = self->GetViewFormat();
    auto indexShape = index->GetViewShape();
    auto indexFormat = index->GetViewFormat();

    // y: same shape/dtype/format as self. sorted_index: follows index dtype (int32/int64),
    // NOT hard-coded INT32 (DESIGN §5.4/§6 correction). Shapes equal the inputs (infershape contract).
    auto values = executor->AllocTensor(selfShape, self->GetDataType(), selfFormat);
    auto sortedIndex = executor->AllocTensor(indexShape, index->GetDataType(), indexFormat);

    L0_DFX(SortWithIndex, self, index, axis, descending, stable, values, sortedIndex);

    ADD_TO_LAUNCHER_LIST_AICORE(
        SortWithIndex, OP_INPUT(self, index), OP_OUTPUT(values, sortedIndex), OP_ATTR(axis, descending, stable));

    return std::tie(values, sortedIndex);
}

} // namespace l0op
