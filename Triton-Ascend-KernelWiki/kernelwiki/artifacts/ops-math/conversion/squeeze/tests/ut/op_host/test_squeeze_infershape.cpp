/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class SqueezeUT : public testing::Test {
};

TEST_F(SqueezeUT, Squeeze_negative_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{-4})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_one_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_two_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 4, 5, 1}, {1, 3, 4, 5, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{0, 4})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_three_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{2, 4, 5})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_unsorted_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{5, 4, 2})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_empty_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_out_of_range_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 5, 1}, {3, 4, 5, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{5})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SqueezeUT, Squeeze_repetive_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{3, 3, 4})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_not_dim1_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Squeeze")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{3, 1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}
