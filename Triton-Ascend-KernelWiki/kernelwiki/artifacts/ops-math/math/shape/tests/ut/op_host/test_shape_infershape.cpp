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

namespace ops {
class ShapeUT : public testing::Test {};

TEST_F(ShapeUT, TensorShape) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Shape")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({4}));
}

TEST_F(ShapeUT, VectorShape) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape")->infer_shape;
  gert::StorageShape input_shape = {{5}, {5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Shape")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({1}));
}

TEST_F(ShapeUT, ScalarShape) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape")->infer_shape;
  gert::StorageShape input_shape = {{}, {}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Shape")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({0}));
}
}  // namespace ops
