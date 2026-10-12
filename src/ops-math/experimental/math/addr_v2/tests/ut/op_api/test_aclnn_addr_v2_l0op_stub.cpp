/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_executor.h"

namespace l0op {

// Stub for l0op::Add（来自 math/add/op_api/add.h，UT 链接时缺少实现）
const aclTensor* Add(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    return executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
}

// Stub for l0op::Mul（来自 math/mul/op_api/mul.h，UT 链接时缺少实现）
const aclTensor* Mul(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    return executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
}

// Stub for l0op::UnsqueezeNd（来自 conversion/unsqueeze/op_host/op_api/unsqueeze.h，UT 链接时缺少实现）
const aclTensor* UnsqueezeNd(const aclTensor* x, const aclIntArray* dim, aclOpExecutor* executor)
{
    return executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
}

// Stub for l0op::LogicalAnd（来自 math/logical_and/op_api/logical_and.h，UT 链接时缺少实现）
const aclTensor* LogicalAnd(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    return executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
}

// Stub for l0op::LogicalOr（来自 math/logical_or/op_api/logical_or.h，UT 链接时缺少实现）
const aclTensor* LogicalOr(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    return executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
}

} // namespace l0op
