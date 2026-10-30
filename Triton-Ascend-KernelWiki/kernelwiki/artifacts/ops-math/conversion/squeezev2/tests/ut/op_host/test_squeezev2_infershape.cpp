/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class SqueezeV2_ut : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SqueezeV2_ut SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "SqueezeV2_ut TearDown" << std::endl;
  }
};

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_negative_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{-4})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_one_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_three_dim) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 1, 4}, {3, 1, 4}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
}

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_two_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 4, 5, 1}, {1, 3, 4, 5, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_three_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_unsorted_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_empty_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_GetUnknownRank_UnknownInputShapeWithEmptyAxis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, -1, 1}, {3, 4, 1, 5, -1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_out_of_range_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 5, 1}, {3, 4, 5, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{4})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 1);
}

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_repetive_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_not_dim1_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{3, 1})
                    .Build();

  EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_unknown_rank_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{-2}, {-2}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axis", std::vector<int64_t>{4})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(SqueezeV2_ut, rt2_SqueezeV2_unknown_dims_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, -1, 5, -1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("SqueezeV2")
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
