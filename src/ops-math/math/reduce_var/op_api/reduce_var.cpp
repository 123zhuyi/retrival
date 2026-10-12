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
 * \file reduce_var.cpp
 * \brief
 */

#include "reduce_var.h"
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
OP_TYPE_REGISTER(ReduceVar);

const std::tuple<const aclTensor *, const aclTensor *> ReduceVar(const aclTensor* self, const aclIntArray* dim,
    int64_t correction, bool keepdim, bool isMeanOut, aclOpExecutor* executor)
{
    L0_DFX(ReduceVar, self, dim, correction, keepdim, isMeanOut);

    aclTensor *reduceVarOut = executor->AllocTensor(self->GetDataType(),
                                                    self->GetStorageFormat(), self->GetOriginalFormat());
    CHECK_RET(reduceVarOut != nullptr, std::tuple(nullptr, nullptr));

    aclTensor *reduceVarMeanOut = executor->AllocTensor(self->GetDataType(),
                                                        self->GetStorageFormat(), self->GetOriginalFormat());
    CHECK_RET(reduceVarMeanOut != nullptr, std::tuple(nullptr, nullptr));

    INFER_SHAPE(ReduceVar, OP_INPUT(self), OP_OUTPUT(reduceVarOut, reduceVarMeanOut),
                OP_ATTR(dim, correction, keepdim, isMeanOut));

    op::Shape outShape = self->GetViewShape();
    auto count = dim->Size();
    size_t dimNum = outShape.GetDimNum();
    if (keepdim) {
        for (uint64_t i = 0; i < count; i++) {
            int64_t dimIndex = static_cast<int64_t>((*dim)[i]);
            int64_t dimNew = dimIndex >= 0 ? dimIndex : dimIndex + dimNum;
            outShape.SetDim(dimNew, 1);
        }
        reduceVarOut->SetViewShape(outShape);
        reduceVarMeanOut->SetViewShape(outShape);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceVar, OP_INPUT(self), OP_OUTPUT(reduceVarOut, reduceVarMeanOut),
                                           OP_ATTR(dim, correction, keepdim, isMeanOut));
    if (ret !=  ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceVar ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }
    return std::tuple(reduceVarOut, reduceVarMeanOut);
}
}  // namespace l0op