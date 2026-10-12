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
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

// 输入顺序与 op_def / proto.h 一致：c, a, b, beta, alpha
// beta/alpha 为 1-element 标量 tensor，不参与 shape 推导
class AddMatMatElementsInfershapeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AddMatMatElementsInfershapeTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AddMatMatElementsInfershapeTest TearDown" << std::endl;
  }
};

TEST_F(AddMatMatElementsInfershapeTest, add_mat_mat_elements_infershape_same_shape) {
  gert::StorageShape shape = {{4, 8}, {4, 8}};
  gert::StorageShape scalarShape = {{1}, {1}};

  gert::InfershapeContextPara infershapeContextPara(
      "AddMatMatElements",
      {{shape, ge::DT_FLOAT, ge::FORMAT_ND},        // c
       {shape, ge::DT_FLOAT, ge::FORMAT_ND},        // a
       {shape, ge::DT_FLOAT, ge::FORMAT_ND},        // b
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND},  // beta
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND}}, // alpha
      {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});

  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddMatMatElementsInfershapeTest, add_mat_mat_elements_infershape_broadcast_a) {
  gert::StorageShape shapeA = {{1, 8}, {1, 8}};
  gert::StorageShape shapeBC = {{4, 8}, {4, 8}};
  gert::StorageShape scalarShape = {{1}, {1}};

  gert::InfershapeContextPara infershapeContextPara(
      "AddMatMatElements",
      {{shapeBC, ge::DT_FLOAT, ge::FORMAT_ND},      // c
       {shapeA, ge::DT_FLOAT, ge::FORMAT_ND},       // a
       {shapeBC, ge::DT_FLOAT, ge::FORMAT_ND},      // b
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND},  // beta
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND}}, // alpha
      {{shapeBC, ge::DT_FLOAT, ge::FORMAT_ND}});

  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddMatMatElementsInfershapeTest, add_mat_mat_elements_infershape_broadcast_b) {
  gert::StorageShape shapeA = {{4, 1}, {4, 1}};
  gert::StorageShape shapeB = {{1, 8}, {1, 8}};
  gert::StorageShape shapeC = {{4, 8}, {4, 8}};
  gert::StorageShape scalarShape = {{1}, {1}};

  gert::InfershapeContextPara infershapeContextPara(
      "AddMatMatElements",
      {{shapeC, ge::DT_FLOAT, ge::FORMAT_ND},       // c
       {shapeA, ge::DT_FLOAT, ge::FORMAT_ND},       // a
       {shapeB, ge::DT_FLOAT, ge::FORMAT_ND},       // b
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND},  // beta
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND}}, // alpha
      {{shapeC, ge::DT_FLOAT, ge::FORMAT_ND}});

  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddMatMatElementsInfershapeTest, add_mat_mat_elements_infershape_3d) {
  gert::StorageShape shape = {{2, 3, 4}, {2, 3, 4}};
  gert::StorageShape scalarShape = {{1}, {1}};

  gert::InfershapeContextPara infershapeContextPara(
      "AddMatMatElements",
      {{shape, ge::DT_FLOAT16, ge::FORMAT_ND},        // c
       {shape, ge::DT_FLOAT16, ge::FORMAT_ND},        // a
       {shape, ge::DT_FLOAT16, ge::FORMAT_ND},        // b
       {scalarShape, ge::DT_FLOAT16, ge::FORMAT_ND},  // beta
       {scalarShape, ge::DT_FLOAT16, ge::FORMAT_ND}}, // alpha
      {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});

  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddMatMatElementsInfershapeTest, add_mat_mat_elements_infershape_broadcast_diff_rank) {
  // a: [8], b: [4, 8], c: [4, 8] -> [4, 8]
  gert::StorageShape shapeA = {{8}, {8}};
  gert::StorageShape shapeBC = {{4, 8}, {4, 8}};
  gert::StorageShape scalarShape = {{1}, {1}};

  gert::InfershapeContextPara infershapeContextPara(
      "AddMatMatElements",
      {{shapeBC, ge::DT_FLOAT, ge::FORMAT_ND},      // c
       {shapeA, ge::DT_FLOAT, ge::FORMAT_ND},       // a
       {shapeBC, ge::DT_FLOAT, ge::FORMAT_ND},      // b
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND},  // beta
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND}}, // alpha
      {{shapeBC, ge::DT_FLOAT, ge::FORMAT_ND}});

  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
