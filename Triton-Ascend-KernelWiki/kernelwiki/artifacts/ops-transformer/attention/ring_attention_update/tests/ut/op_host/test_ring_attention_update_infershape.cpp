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
 * \file test_ring_attention_update_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RingAttentionUpdate : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RingAttentionUpdate SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RingAttentionUpdate TearDown" << std::endl;
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

static void ExeTestCase(
    std::vector<std::vector<int64_t> > expectResults,
    const gert::StorageShape& prevAttnOutShape,
    const gert::StorageShape& prevSoftmaxMaxShape,
    const gert::StorageShape& prevSoftmaxSumShape,
    const gert::StorageShape& curAttnOutShape,
    const gert::StorageShape& curSoftmaxMaxShape,
    const gert::StorageShape& curSoftmaxSumShape,
    const ge::DataType& attnOutDtype,
    const ge::DataType& softmaxDtype,
    gert::StorageShape& attnOutShape,
    gert::StorageShape& softmaxMaxShape,
    gert::StorageShape& softmaxSumShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&prevAttnOutShape,
        (gert::Tensor *)&prevSoftmaxMaxShape,
        (gert::Tensor *)&prevSoftmaxSumShape,
        (gert::Tensor *)&curAttnOutShape,
        (gert::Tensor *)&curSoftmaxMaxShape,
        (gert::Tensor *)&curSoftmaxSumShape
    };
    std::vector<gert::StorageShape *> outputShapes = {
        &attnOutShape,
        &softmaxMaxShape,
        &softmaxSumShape,
    };
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("RingAttentionUpdate")
        .NodeIoNum(6, 3)
        .NodeInputTd(0, attnOutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, softmaxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, softmaxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, attnOutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, softmaxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, softmaxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, attnOutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, softmaxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, softmaxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("RingAttentionUpdate")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(RingAttentionUpdate, RingAttentionUpdate_infershape_test_0) {
  size_t sequence = 1024;
  size_t batch = 2;
  size_t headDim = 384;
  size_t headNum = 3;
  size_t lastDim = 8;
  gert::StorageShape prevAttnOutShape = {{sequence, batch, headDim}, {sequence, batch, headDim}};
  gert::StorageShape prevSoftmaxMaxShape = {{batch, headNum, sequence, lastDim}, {batch, headNum, sequence, lastDim}};
  gert::StorageShape prevSoftmaxSumShape = {{batch, headNum, sequence, lastDim}, {batch, headNum, sequence, lastDim}};
  gert::StorageShape curAttnOutShape = {{sequence, batch, headDim}, {sequence, batch, headDim}};
  gert::StorageShape curSoftmaxMaxShape = {{batch, headNum, sequence, lastDim}, {batch, headNum, sequence, lastDim}};
  gert::StorageShape curSoftmaxSumShape = {{batch, headNum, sequence, lastDim}, {batch, headNum, sequence, lastDim}};
  gert::StorageShape attnOutShape = {};
  gert::StorageShape softmaxMaxShape = {};
  gert::StorageShape softmaxSumShape = {};
  std::vector<int64_t> expectResult0 = {sequence, batch, headDim};
  std::vector<int64_t> expectResult1 = {batch, headNum, sequence, lastDim};
  std::vector<int64_t> expectResult2 = {batch, headNum, sequence, lastDim};
  ExeTestCase({expectResult0,expectResult1,expectResult2}, prevAttnOutShape, prevSoftmaxMaxShape, prevSoftmaxSumShape,
              curAttnOutShape, curSoftmaxMaxShape, curSoftmaxSumShape,
              ge::DT_FLOAT, ge::DT_FLOAT,
              attnOutShape, softmaxMaxShape, softmaxSumShape, ge::GRAPH_SUCCESS);
}