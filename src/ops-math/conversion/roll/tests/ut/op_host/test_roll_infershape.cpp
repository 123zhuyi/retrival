/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RollUT : public testing::Test {};

TEST_F(RollUT, Roll_2d_tensor_with_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{4, 3}, {4, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{1})
                      .Attr("dims", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

TEST_F(RollUT, Roll_3d_tensor_with_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{2})
                      .Attr("dims", std::vector<int64_t>{2})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
}

TEST_F(RollUT, Roll_1d_tensor_without_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{6}, {6}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{2})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 6);
}

TEST_F(RollUT, Roll_empty_dims)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{4, 3}, {4, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{2})
                      .Attr("dims", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

TEST_F(RollUT, Roll_multiple_shifts_dims)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{1, 2})
                      .Attr("dims", std::vector<int64_t>{0, 2})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
}

TEST_F(RollUT, Roll_negative_shift)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{4, 3}, {4, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{-1})
                      .Attr("dims", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

TEST_F(RollUT, Roll_negative_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{4, 3}, {4, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{1})
                      .Attr("dims", std::vector<int64_t>{-1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

TEST_F(RollUT, Roll_4d_tensor)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{1})
                      .Attr("dims", std::vector<int64_t>{1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

TEST_F(RollUT, Roll_large_shift)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{4, 3}, {4, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{10})
                      .Attr("dims", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

TEST_F(RollUT, Roll_single_element)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{1}, {1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{1})
                      .Attr("dims", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
}

TEST_F(RollUT, Roll_5d_tensor)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Roll")->infer_shape;
    gert::StorageShape input_shape = {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Roll")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shifts", std::vector<int64_t>{1})
                      .Attr("dims", std::vector<int64_t>{4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 5);
}
