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
#include <vector>

#include "opdev/make_op_executor.h"
#include "../../../op_host/op_api/triangular_solve.h"

const int64_t DATA_SIZE = 1024 * 1024;

class TriangularSolveTest: public ::testing::Test {
 public:
  TriangularSolveTest() : l0Executor(nullptr) {}

  aclTensor *CreateContiguousAclTensor(std::vector<int64_t> viewShape, aclDataType dtype) {
    std::vector<int64_t> stride(viewShape.size(), 1);
    for (int i = viewShape.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * viewShape[i];
    }
    return aclCreateTensor(viewShape.data(), viewShape.size(), dtype, stride.data(), 0,
                           ACL_FORMAT_ND, viewShape.data(), viewShape.size(), data);
  }

  void Clear() {
  }

  void SetUp() override {
    auto l2Executor = &l0Executor;
    auto uniqueExecutor = CREATE_EXECUTOR();
    uniqueExecutor.ReleaseTo(l2Executor);
  }

  void TearDown() override {
    delete l0Executor;
  }

 public:
  aclOpExecutor* l0Executor;
  int64_t data[DATA_SIZE] = {0};
};

TEST_F(TriangularSolveTest, TriangularSolve_float_upper_false) {
  auto self = CreateContiguousAclTensor({3, 4}, ACL_FLOAT);
  auto A = CreateContiguousAclTensor({3, 3}, ACL_FLOAT);
  auto xOut = CreateContiguousAclTensor({3, 4}, ACL_FLOAT);
  bool upper = false;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_float_upper_true) {
  auto self = CreateContiguousAclTensor({3, 4}, ACL_FLOAT);
  auto A = CreateContiguousAclTensor({3, 3}, ACL_FLOAT);
  auto xOut = CreateContiguousAclTensor({3, 4}, ACL_FLOAT);
  bool upper = true;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_transpose_true) {
  auto self = CreateContiguousAclTensor({3, 4}, ACL_FLOAT);
  auto A = CreateContiguousAclTensor({3, 3}, ACL_FLOAT);
  auto xOut = CreateContiguousAclTensor({3, 4}, ACL_FLOAT);
  bool upper = true;
  bool transpose = true;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_double) {
  auto self = CreateContiguousAclTensor({3, 4}, ACL_DOUBLE);
  auto A = CreateContiguousAclTensor({3, 3}, ACL_DOUBLE);
  auto xOut = CreateContiguousAclTensor({3, 4}, ACL_DOUBLE);
  bool upper = true;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_complex64) {
  auto self = CreateContiguousAclTensor({3, 4}, ACL_COMPLEX64);
  auto A = CreateContiguousAclTensor({3, 3}, ACL_COMPLEX64);
  auto xOut = CreateContiguousAclTensor({3, 4}, ACL_COMPLEX64);
  bool upper = true;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_complex128) {
  auto self = CreateContiguousAclTensor({3, 4}, ACL_COMPLEX128);
  auto A = CreateContiguousAclTensor({3, 3}, ACL_COMPLEX128);
  auto xOut = CreateContiguousAclTensor({3, 4}, ACL_COMPLEX128);
  bool upper = true;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_batch) {
  auto self = CreateContiguousAclTensor({2, 3, 4}, ACL_FLOAT);
  auto A = CreateContiguousAclTensor({2, 3, 3}, ACL_FLOAT);
  auto xOut = CreateContiguousAclTensor({2, 3, 4}, ACL_FLOAT);
  bool upper = true;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({2, 3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_4d_batch) {
  auto self = CreateContiguousAclTensor({1, 1, 3, 4}, ACL_FLOAT);
  auto A = CreateContiguousAclTensor({1, 1, 3, 3}, ACL_FLOAT);
  auto xOut = CreateContiguousAclTensor({1, 1, 3, 4}, ACL_FLOAT);
  bool upper = true;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({1, 1, 3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_single_col) {
  auto self = CreateContiguousAclTensor({3, 1}, ACL_FLOAT);
  auto A = CreateContiguousAclTensor({3, 3}, ACL_FLOAT);
  auto xOut = CreateContiguousAclTensor({3, 1}, ACL_FLOAT);
  bool upper = true;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({3, 1});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

TEST_F(TriangularSolveTest, TriangularSolve_empty_batch) {
  auto self = CreateContiguousAclTensor({0, 3, 4}, ACL_FLOAT);
  auto A = CreateContiguousAclTensor({0, 3, 3}, ACL_FLOAT);
  auto xOut = CreateContiguousAclTensor({0, 3, 4}, ACL_FLOAT);
  bool upper = true;
  bool transpose = false;
  auto result = l0op::TriangularSolve(self, A, upper, transpose, xOut, l0Executor);
  ASSERT_NE(result, nullptr);

  op::ShapeVector expectShape({0, 3, 4});
  EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}