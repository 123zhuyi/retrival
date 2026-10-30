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

class UnsqueezeV2_ut : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UnsqueezeV2_ut SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "UnsqueezeV2_ut TearDown" << std::endl;
  }
};

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_negative_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{-5})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 2);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 5);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_one_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_two_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{0, 4})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 1);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_three_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{2, 4, 5})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 6);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(5), 1);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_unsorted_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{5, 4, 2})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 6);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(5), 1);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_empty_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_out_of_range_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{4})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_repetive_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{1, 1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_larger_than_kMaxDimNums_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5, 1, 1, 1, 1}, {3, 4, 5, 1, 1, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{1, 2, 3, 4, 5})
                    .Build();

  // EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UnsqueezeV2_ut, rt2_Unsqueeze_unknown_rank_axes) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;

  gert::StorageShape input_shape = {{-2}, {-2}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV2")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Attr("axes", std::vector<int64_t>{1, 2, 3, 4, 5})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}