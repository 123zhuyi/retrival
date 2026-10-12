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
 * \file test_cummin_infershape.cpp
 * \brief Cummin InferShape and InferDataType UT.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class CumminInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CumminInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CumminInfershape TearDown" << std::endl; }
};

// 输出 y、argmin 的 shape 均与输入 x 一致
TEST_F(CumminInfershape, cummin_infer_shape_same)
{
    gert::InfershapeContextPara infershapeContextPara("Cummin",
                                                      {
                                                          {{{3, 2}, {3, 2}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 2},
        {3, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 标量输入
TEST_F(CumminInfershape, cummin_infer_shape_scalar)
{
    gert::InfershapeContextPara infershapeContextPara("Cummin",
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态维度（-1）
TEST_F(CumminInfershape, cummin_infer_shape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("Cummin",
                                                      {
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知 rank（-2）
TEST_F(CumminInfershape, cummin_infer_shape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("Cummin",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// InferDataType：y.dtype 与 x 一致，argmin.dtype 固定为 int32（float32）
TEST_F(CumminInfershape, cummin_infer_datatype_float)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Cummin");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Cummin").OpName("Cummin");
    builder.IONum(1, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

// InferDataType：y.dtype 与 x 一致，argmin.dtype 固定为 int32（float16）
TEST_F(CumminInfershape, cummin_infer_datatype_float16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Cummin");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Cummin").OpName("Cummin");
    builder.IONum(1, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

// InferDataType：y.dtype 与 x 一致，argmin.dtype 固定为 int32（bf16）
TEST_F(CumminInfershape, cummin_infer_datatype_bf16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Cummin");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Cummin").OpName("Cummin");
    builder.IONum(1, 2);
    builder.InputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

// InferDataType：y.dtype 与 x 一致，argmin.dtype 固定为 int32（int32）
TEST_F(CumminInfershape, cummin_infer_datatype_int32)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Cummin");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Cummin").OpName("Cummin");
    builder.IONum(1, 2);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}
