/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_atan2_infershape.cpp
 * \brief Atan2 InferShape and InferDataType UT.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class Atan2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Atan2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Atan2Infershape TearDown" << std::endl; }
};

// 输入 shape 相同，输出 shape 与输入一致
TEST_F(Atan2Infershape, atan2_infer_shape_same)
{
    gert::InfershapeContextPara infershapeContextPara("Atan2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 第二个输入为标量，输出 shape 与第一个输入一致
TEST_F(Atan2Infershape, atan2_infer_shape_broadcast_scalar)
{
    gert::InfershapeContextPara infershapeContextPara("Atan2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 一般广播：{3, 1, 4} 与 {2, 1} 广播为 {3, 2, 4}
TEST_F(Atan2Infershape, atan2_infer_shape_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara("Atan2",
                                                      {
                                                          {{{3, 1, 4}, {3, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 1}, {2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 2, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态维度（-1）
TEST_F(Atan2Infershape, atan2_infer_shape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("Atan2",
                                                      {
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知 rank（-2）
TEST_F(Atan2Infershape, atan2_infer_shape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("Atan2",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// kernel 不支持混合 dtype，两个输入 dtype 不同时推断失败
TEST_F(Atan2Infershape, atan2_infer_shape_mixed_dtype_rejected)
{
    gert::InfershapeContextPara infershapeContextPara("Atan2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// InferDataType：输出 dtype 与输入 x1 一致（float32）
TEST_F(Atan2Infershape, atan2_infer_datatype_float)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Atan2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Atan2").OpName("Atan2");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

// InferDataType：输出 dtype 与输入 x1 一致（float16）
TEST_F(Atan2Infershape, atan2_infer_datatype_float16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Atan2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Atan2").OpName("Atan2");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// InferDataType：输出 dtype 与输入 x1 一致（bf16）
TEST_F(Atan2Infershape, atan2_infer_datatype_bf16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Atan2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Atan2").OpName("Atan2");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

// InferDataType：两个输入 dtype 不同时，输出 dtype 取 x1 的 dtype
TEST_F(Atan2Infershape, atan2_infer_datatype_mixed_input)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Atan2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Atan2").OpName("Atan2");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

// InferDataType：GE 层输出 dtype 取 x1，不做类型提升
// （x1=float16、x2=float32 时输出仍为 float16；提升语义由 aclnn API 层负责）
TEST_F(Atan2Infershape, atan2_infer_datatype_follows_x1)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Atan2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Atan2").OpName("Atan2");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
