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

class Rank_UT : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "Rank_UT SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "Rank_UT TearDown" << std::endl;
    }
};

TEST_F(Rank_UT, InferShape_succ) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Rank"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Rank")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("Rank")
                    .NodeIoNum(1, 1)
                    .InputTensors({(gert::Tensor *)&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().IsScalar(), true);  
}