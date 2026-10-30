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
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

using std::make_pair;
class SHAPE_SHAPEN_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SHAPE_SHAPEN_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SHAPE_SHAPEN_UT TearDown" << std::endl;
  }
};

TEST_F(SHAPE_SHAPEN_UT, TensorShapeRT2) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN")->infer_shape;
  gert::StorageShape input_shape_0 = {{1, 3, 4, 5}, {1, 3, 4, 5}};
  gert::StorageShape input_shape_1 = {{1, 3, 4}, {1, 3, 4}};
  gert::StorageShape output_shape_0 = {{}, {}};
  gert::StorageShape output_shape_1 = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("ShapeN")
                    .NodeIoNum(2, 2)
                    .InputTensors({(gert::Tensor *)&input_shape_0, (gert::Tensor *)&input_shape_1})
                    .OutputShapes({&output_shape_0, &output_shape_1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({4}));
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(1)), gert::Shape({3}));
}

TEST_F(SHAPE_SHAPEN_UT, VectorShapeNRT2) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN")->infer_shape;
  gert::StorageShape input_shape_0 = {{5}, {5}};
  gert::StorageShape input_shape_1 = {{4}, {4}};
  gert::StorageShape output_shape_0 = {{}, {}};
  gert::StorageShape output_shape_1 = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("ShapeN")
                    .NodeIoNum(2, 2)
                    .InputTensors({(gert::Tensor *)&input_shape_0, (gert::Tensor *)&input_shape_1})
                    .OutputShapes({&output_shape_0, &output_shape_1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({1}));
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(1)), gert::Shape({1}));
}

TEST_F(SHAPE_SHAPEN_UT, ScalarShapeN) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN")->infer_shape;
  gert::StorageShape input_shape_0 = {{}, {}};
  gert::StorageShape input_shape_1 = {{}, {}};
  gert::StorageShape output_shape_0 = {{}, {}};
  gert::StorageShape output_shape_1 = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("ShapeN")
                    .NodeIoNum(2, 2)
                    .InputTensors({(gert::Tensor *)&input_shape_0, (gert::Tensor *)&input_shape_1})
                    .OutputShapes({&output_shape_0, &output_shape_1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({0}));
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(1)), gert::Shape({0}));
}