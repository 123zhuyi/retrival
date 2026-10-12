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
 * \file adjacent_difference.cpp
 * \brief
 */
#include "adjacent_difference.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op
{
OP_TYPE_REGISTER(AdjacentDifference);

aclTensor* AdjacentDifference(const aclTensor *x, op::DataType yDtype, aclOpExecutor *executor)
{
    OP_LOGI("AdjacentDifference L0 Start.");
    L0_DFX(AdjacentDifference, x, yDtype);

    auto y = executor->AllocTensor(x->GetViewShape(), yDtype);
    ADD_TO_LAUNCHER_LIST_AICORE(AdjacentDifference, OP_INPUT(x), OP_OUTPUT(y), OP_ATTR(yDtype));
    OP_LOGI("AdjacentDifference Launch finish.");
    return y;
}

}  // namespace l0op
