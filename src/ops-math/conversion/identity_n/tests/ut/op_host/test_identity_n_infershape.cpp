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

class IdentityNRTTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Identity SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Identity TearDown" << std::endl;
  }
};

TEST_F(IdentityNRTTest, identityN_infer_shape_known_success) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("IdentityN"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("IdentityN")->infer_shape;
  gert::StorageShape input_shape_0 = {{1, 3, 4, 5}, {1, 3, 4, 5}};
  gert::StorageShape input_shape_1 = {{2, 3, 4, 5}, {2, 3, 4, 5}};
  gert::StorageShape output_shape_0 = {{}, {}};
  gert::StorageShape output_shape_1 = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("IdentityN")
                    .NodeIoNum(2, 2)
                    .InputTensors({(gert::Tensor *)&input_shape_0, (gert::Tensor *)&input_shape_1})
                    .OutputShapes({&output_shape_0, &output_shape_1})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({1, 3, 4, 5}));
  EXPECT_EQ(*(holder.GetContext()->GetOutputShape(1)), gert::Shape({2, 3, 4, 5}));
}