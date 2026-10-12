/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moeTokenUnpermuteWithEpGrad_infershape.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
class MoeTokenUnpermuteWithEpGradInferShape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpGrad Proto Test TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_infershape_case_0)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
     gert::StorageShape blank_shape = {};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    auto holder =
        gert::InferShapeContextFaker()
            .SetOpType("MoeTokenUnpermuteWithEpGrad")
            .NodeIoNum(4, 2)
            .OutputShapes({&permuted_tokens_grad_shape, &probs_grad_shape})
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputTensors({(gert::Tensor *)&unpermuted_output_d_shape})
            .InputTensors({(gert::Tensor *)&sorted_indices_shape})
            .InputTensors({(gert::Tensor *)&permuted_tokens_shape})
            .InputTensors({(gert::Tensor *)&probs_shape})
            .Attr("padded_mode", bool(false))
            .Build();

    gert::InferShapeContext* context = holder.GetContext();
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedPermutedTokensGradShape = {30, 64};
    std::vector<int64_t> expectedProbsGradShape = {10, 3};
    auto permutedTokensGradShape = context->GetOutputShape(0);
    auto probsGradShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*permutedTokensGradShape), expectedPermutedTokensGradShape);
    EXPECT_EQ(ToVector(*probsGradShape), expectedProbsGradShape);
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_infershape_case_01)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    auto holder =
        gert::InferShapeContextFaker()
            .SetOpType("MoeTokenUnpermuteWithEpGrad")
            .NodeIoNum(4, 2)
            .OutputShapes({&permuted_tokens_grad_shape, &probs_grad_shape})
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputTensors({(gert::Tensor *)&unpermuted_output_d_shape})
            .InputTensors({(gert::Tensor *)&sorted_indices_shape})
            .InputTensors({(gert::Tensor *)&permuted_tokens_shape})
            .InputTensors({(gert::Tensor *)&probs_shape})
            .Attr("padded_mode", bool(false))
            .Build();

    gert::InferShapeContext* context = holder.GetContext();
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedPermutedTokensGradShape = {30, 64};
    std::vector<int64_t> expectedProbsGradShape = {10, 3};
    auto permutedTokensGradShape = context->GetOutputShape(0);
    auto probsGradShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*permutedTokensGradShape), expectedPermutedTokensGradShape);
    EXPECT_EQ(ToVector(*probsGradShape), expectedProbsGradShape);
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_infershape_case_1)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    auto holder =
        gert::InferShapeContextFaker()
            .SetOpType("MoeTokenUnpermuteWithEpGrad")
            .NodeIoNum(4, 2)
            .OutputShapes({&permuted_tokens_grad_shape, &probs_grad_shape})
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputTensors({(gert::Tensor *)&unpermuted_output_d_shape})
            .InputTensors({(gert::Tensor *)&sorted_indices_shape})
            .InputTensors({(gert::Tensor *)&permuted_tokens_shape})
            .InputTensors({(gert::Tensor *)&probs_shape})
            .Attr("padded_mode", bool(false))
            .Build();

    gert::InferShapeContext* context = holder.GetContext();
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedPermutedTokensGradShape = {30, 64};
    std::vector<int64_t> expectedProbsGradShape = {10, 3};
    auto permutedTokensGradShape = context->GetOutputShape(0);
    auto probsGradShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*permutedTokensGradShape), expectedPermutedTokensGradShape);
    EXPECT_EQ(ToVector(*probsGradShape), expectedProbsGradShape);
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_inferdtype_bf16_case_0)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_indices_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_inferdtype_bf16_case_1)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_ref_prob = ge::DT_FLOAT;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_BF16;
        ge::DataType output_ref_prob = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_indices_ref, &input_ref, &input_ref_prob})
                                  .OutputDataTypes({&output_ref, &output_ref_prob})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_prob);
    }
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_infershape_prob_not_none_split_h_case_0)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    auto holder =
        gert::InferShapeContextFaker()
            .SetOpType("MoeTokenUnpermuteWithEpGrad")
            .NodeIoNum(4, 2)
            .OutputShapes({&permuted_tokens_grad_shape, &probs_grad_shape})
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputTensors({(gert::Tensor *)&unpermuted_output_d_shape})
            .InputTensors({(gert::Tensor *)&sorted_indices_shape})
            .InputTensors({(gert::Tensor *)&permuted_tokens_shape})
            .InputTensors({(gert::Tensor *)&probs_shape})
            .Attr("padded_mode", bool(false))
            .Build();

    gert::InferShapeContext* context = holder.GetContext();
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedPermutedTokensGradShape = {30, 8192};
    std::vector<int64_t> expectedProbsGradShape = {10, 3};
    auto permutedTokensGradShape = context->GetOutputShape(0);
    auto probsGradShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*permutedTokensGradShape), expectedPermutedTokensGradShape);
    EXPECT_EQ(ToVector(*probsGradShape), expectedProbsGradShape);
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_inferdtype_fp16_case_0)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType input_indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_indices_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_infershape_prob_not_none_split_h_fp16_case_0)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    auto holder =
        gert::InferShapeContextFaker()
            .SetOpType("MoeTokenUnpermuteWithEpGrad")
            .NodeIoNum(4, 2)
            .OutputShapes({&permuted_tokens_grad_shape, &probs_grad_shape})
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputTensors({(gert::Tensor *)&unpermuted_output_d_shape})
            .InputTensors({(gert::Tensor *)&sorted_indices_shape})
            .InputTensors({(gert::Tensor *)&permuted_tokens_shape})
            .InputTensors({(gert::Tensor *)&probs_shape})
            .Attr("padded_mode", bool(false))
            .Build();

    gert::InferShapeContext* context = holder.GetContext();
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedPermutedTokensGradShape = {30, 8192};
    std::vector<int64_t> expectedProbsGradShape = {10, 3};
    auto permutedTokensGradShape = context->GetOutputShape(0);
    auto probsGradShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*permutedTokensGradShape), expectedPermutedTokensGradShape);
    EXPECT_EQ(ToVector(*probsGradShape), expectedProbsGradShape);
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_infershape_prob_not_none_split_h_fp16_case_1)
{
    gert::StorageShape permuted_tokens_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape unpermuted_output_d_shape = {{10, 8192}, {10, 8192}};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 8192}, {30, 8192}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    auto holder =
        gert::InferShapeContextFaker()
            .SetOpType("MoeTokenUnpermuteWithEpGrad")
            .NodeIoNum(4, 2)
            .OutputShapes({&permuted_tokens_grad_shape, &probs_grad_shape})
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputTensors({(gert::Tensor *)&unpermuted_output_d_shape})
            .InputTensors({(gert::Tensor *)&sorted_indices_shape})
            .InputTensors({(gert::Tensor *)&permuted_tokens_shape})
            .InputTensors({(gert::Tensor *)&probs_shape})
            .Attr("padded_mode", bool(false))
            .Build();

    gert::InferShapeContext* context = holder.GetContext();
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedPermutedTokensGradShape = {30, 8192};
    std::vector<int64_t> expectedProbsGradShape = {10, 3};
    auto permutedTokensGradShape = context->GetOutputShape(0);
    auto probsGradShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*permutedTokensGradShape), expectedPermutedTokensGradShape);
    EXPECT_EQ(ToVector(*probsGradShape), expectedProbsGradShape);
}
