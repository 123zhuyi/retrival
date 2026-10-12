/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "infershape_context_faker.h"

TEST(CaseConditionInferShape, SetsScalarOutput)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto* opImpl = registry->GetOpImpl("CaseCondition");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);

    gert::StorageShape inputShape = {{3}, {3}};
    gert::StorageShape outputShape = {{1}, {1}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("CaseCondition")
                      .NodeIoNum(1, 1)
                      .InputTensors({reinterpret_cast<gert::Tensor*>(&inputShape)})
                      .OutputShapes({&outputShape})
                      .Build();

    auto* context = holder.GetContext();
    EXPECT_EQ(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    EXPECT_TRUE(context->GetOutputShape(0)->IsScalar());
}
