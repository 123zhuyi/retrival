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

#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {

ge::graphStatus RunInferShape(const gert::Shape& inputShape, gert::Shape& outputShape, bool withSelfShape = true)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftshrinkGrad");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape gradShape = inputShape;
    gert::Shape xShape = inputShape;
    std::vector<gert::Shape*> inputShapes = {&gradShape, withSelfShape ? &xShape : nullptr};
    std::vector<gert::Shape*> outputShapes = {&outputShape};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SoftshrinkGrad")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(inputShapes)
                      .OutputShapes(outputShapes)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        const gert::Shape* inferredShape = context->GetOutputShape(0);
        if (inferredShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        outputShape = *inferredShape;
    }
    return status;
}

TEST(SoftShrinkGradInferShapeTest, CopiesDynamicInputShape)
{
    gert::Shape inputShape({1, -1, 32, 17});
    gert::Shape outputShape;
    ASSERT_EQ(RunInferShape(inputShape, outputShape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(outputShape.GetDimNum(), inputShape.GetDimNum());
    for (size_t i = 0; i < inputShape.GetDimNum(); ++i) {
        EXPECT_EQ(outputShape.GetDim(i), inputShape.GetDim(i));
    }
}

TEST(SoftShrinkGradInferShapeTest, CopiesScalarShape)
{
    gert::Shape inputShape;
    gert::Shape outputShape({1});
    ASSERT_EQ(RunInferShape(inputShape, outputShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputShape.GetDimNum(), 0U);
}

TEST(SoftShrinkGradInferShapeTest, RejectsNullSelfShape)
{
    gert::Shape inputShape({2, 3});
    gert::Shape outputShape;
    EXPECT_EQ(RunInferShape(inputShape, outputShape, false), ge::GRAPH_FAILED);
}

} // namespace
