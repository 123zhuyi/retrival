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
 * \file test_stft_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class STFT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "STFT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "STFT TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}


static void ExeTestCase(std::vector<std::vector<int64_t> > expectResults,
                        const gert::StorageShape& xShape,
                        const gert::StorageShape& windowShape,
                        gert::StorageShape& outputShape,
                        ge::DataType inputDtype,
                        ge::DataType windowDtype,
                        ge::DataType outputDtype,
                        int64_t hopLength,
                        int64_t winLength,
                        bool normalized,
                        bool onesided,
                        bool returnComplex,
                        int64_t nFft,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&xShape, (gert::Tensor*)&xShape, (gert::Tensor*)&windowShape};
    std::vector<gert::StorageShape *> outputShapes = {&outputShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("STFT")
        .NodeIoNum(3, 1)
        .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, windowDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Attr("hop_length", hopLength)
        .Attr("win_length", winLength)
        .Attr("normalized", normalized)
        .Attr("onesided", onesided)
        .Attr("return_complex", returnComplex)
        .Attr("n_fft", nFft)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

static void ExeTestCaseNoNFft(const gert::StorageShape& xShape,
                        const gert::StorageShape& windowShape,
                        gert::StorageShape& outputShape,
                        ge::DataType inputDtype,
                        ge::DataType windowDtype,
                        ge::DataType outputDtype,
                        int64_t hopLength,
                        int64_t winLength,
                        bool normalized,
                        bool onesided,
                        bool returnComplex,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&xShape, (gert::Tensor*)&xShape, (gert::Tensor*)&windowShape};
    std::vector<gert::StorageShape *> outputShapes = {&outputShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("STFT")
        .NodeIoNum(3, 1)
        .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, windowDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Attr("hop_length", hopLength)
        .Attr("win_length", winLength)
        .Attr("normalized", normalized)
        .Attr("onesided", onesided)
        .Attr("return_complex", returnComplex)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("STFT")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
}

TEST_F(STFT, STFT_FP32_IN_1D_NO_WIN_ONESIDED_RETURN_REAL)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    std::vector<int64_t> expectResult = {2, 3, 2};
    ExeTestCase({expectResult}, 
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_CFP128_IN_1D_NO_WIN_TWOSIDED_RETURN_COMPLEX)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = true;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_COMPLEX64;
    std::vector<int64_t> expectResult = {3, 3};
    ExeTestCase({expectResult}, 
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_FP64_IN_2D_NO_WIN_ONESIDED_RETURN_REAL)
{
    gert::StorageShape xShape = {{1, 10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    std::vector<int64_t> expectResult = {1, 2, 3, 2};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_CFP64_IN_2D_NO_WIN_TWOSIDED_RETURN_COMPLEX)
{
    gert::StorageShape xShape = {{1, 10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = true;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_COMPLEX64;
    std::vector<int64_t> expectResult = {1, 3, 3};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_FP64_IN_CFP128_WIN_RETURN_COMPLEX)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = true;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_COMPLEX64;
    std::vector<int64_t> expectResult = {3, 3};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

TEST_F(STFT, STFT_FP64_IN_CFP128_WIN_RETURN_REAL)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    std::vector<int64_t> expectResult = {3, 3, 2};
    ExeTestCase({expectResult},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_SUCCESS);
}

// exception instance
TEST_F(STFT, STFT_NO_N_FFT)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    ExeTestCaseNoNFft(
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_INPUT_NOT_1D_2D)
{
    gert::StorageShape xShape = {{1, 1, 10}, {}};
    gert::StorageShape windowShape = {{1}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    ExeTestCase({},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_WINDOWS_NOT_1D)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{1, 3}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    ExeTestCase({},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_FAILED);
}

TEST_F(STFT, STFT_WINDOWS_LEN_NOT_MATCH)
{
    gert::StorageShape xShape = {{10}, {}};
    gert::StorageShape windowShape = {{4}, {}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t hopLength = 3;
    int64_t winLength = 3;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    int64_t nFft = 3;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType windowDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    ExeTestCase({},
        xShape, windowShape, outputShape, inputDtype, windowDtype, outputDtype, hopLength, winLength, normalized,
        onesided, returnComplex, nFft, ge::GRAPH_FAILED);
}

// TEST_F(STFT, STFT_INPUT_WRONG_TYPE)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_x_ref = ge::DT_FLOAT16;
//     ge::DataType input_win_ref = ge::DT_COMPLEX128;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
// }

// TEST_F(STFT, STFT_WINDOW_WRONG_TYPE)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_x_ref = ge::DT_FLOAT;
//     ge::DataType input_win_ref = ge::DT_FLOAT16;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
// }

// TEST_F(STFT, STFT_DOUBLE_C128_INFER_DATATYPE_RETURN_DT_DOUBLE)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_x_ref = ge::DT_DOUBLE;
//     ge::DataType input_win_ref = ge::DT_COMPLEX128;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_DOUBLE);
// }

// TEST_F(STFT, STFT_FP32_C64_INFER_DATATYPE_RETURN_DT_FP32)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_x_ref = ge::DT_FLOAT;
//     ge::DataType input_win_ref = ge::DT_COMPLEX64;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT);
// }

// TEST_F(STFT, STFT_FP32_C128_INFER_DATATYPE_RETURN_DT_DOUBLE)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_x_ref = ge::DT_FLOAT;
//     ge::DataType input_win_ref = ge::DT_COMPLEX128;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_DOUBLE);
// }

// TEST_F(STFT, STFT_DOUBLE_C128_INFER_DATATYPE_RETURN_C128)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_x_ref = ge::DT_DOUBLE;
//     ge::DataType input_win_ref = ge::DT_COMPLEX128;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX128);
// }

// TEST_F(STFT, STFT_FP32_C64_INFER_DATATYPE_RETURN_C64)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_x_ref = ge::DT_FLOAT;
//     ge::DataType input_win_ref = ge::DT_COMPLEX64;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX64);
// }

// TEST_F(STFT, STFT_FP32_C128_INFER_DATATYPE_RETURN_C128)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_x_ref = ge::DT_FLOAT;
//     ge::DataType input_win_ref = ge::DT_COMPLEX128;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_x_ref, &input_x_ref, &input_win_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX128);
// }

// TEST_F(STFT, STFT_C64_INFER_DATATYPE_RETURN_C64)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_ref = ge::DT_COMPLEX64;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_ref, &input_ref, &input_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX64);
// }

// TEST_F(STFT, STFT_DOUBLE_INFER_DATATYPE_RETURN_DOUBLE)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_ref = ge::DT_DOUBLE;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_ref, &input_ref, &input_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_DOUBLE);
// }

// TEST_F(STFT, STFT_C128_INFER_DATATYPE_RETURN_C128)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_ref = ge::DT_COMPLEX128;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_ref, &input_ref, &input_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX128);
// }

// TEST_F(STFT, STFT_FP32_INFER_DATATYPE_RETURN_FP32)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_ref = ge::DT_FLOAT;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_ref, &input_ref, &input_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT);
// }

// TEST_F(STFT, STFT_FP32_INFER_DATATYPE_RETURN_COMPLEX64)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("STFT"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("STFT")->infer_datatype;
//     ASSERT_NE(data_type_func, nullptr);
//     ge::DataType input_ref = ge::DT_FLOAT;
//     ge::DataType output_ref = ge::DT_FLOAT16;

//     gert::StorageShape outputShape = {{}, {}};
//     auto context_holder = gert::InferDataTypeContextFaker()
//                               .NodeIoNum(3, 1)
//                               .IrInstanceNum({1, 1, 1})
//                               .InputDataTypes({&input_ref, &input_ref, &input_ref})
//                               .OutputDataTypes({&output_ref})
//                               .NodeAttrs(
//                                   {{"hop_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"win_length", ge::AnyValue::CreateFrom<int64_t>(3)},
//                                    {"normalized", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"onesided", ge::AnyValue::CreateFrom<bool>(false)},
//                                    {"return_complex", ge::AnyValue::CreateFrom<bool>(true)},
//                                    {"n_fft", ge::AnyValue::CreateFrom<int64_t>(3)}})
//                               .Build();

//     EXPECT_EQ(data_type_func(context_holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context_holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_COMPLEX64);
// }