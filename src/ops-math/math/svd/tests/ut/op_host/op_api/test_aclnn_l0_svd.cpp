/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "opdev/make_op_executor.h"
#include "math/svd/op_host/op_api/svd.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 6;

class SvdTest : public ::testing::Test {
 public:
 SvdTest() : exe(nullptr) {
  }

  aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype) {
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           data);
  }

  void Clear() {
  }

  void SetUp() override {
    auto executor = &exe;
    auto unique_executor = CREATE_EXECUTOR();
    unique_executor.ReleaseTo(executor);
  }

  void TearDown() override {
    delete exe;
  }

 public:
  aclOpExecutor* exe;
  int64_t data[DATA_SIZE] = {1};
};

TEST_F(SvdTest, SvdTest_fullMatrices_true_SUCC) {
  auto x = CreateAclTensor({2, 3}, ACL_FLOAT);
  bool fullMatrices = true;
  bool computeUV= true;
  auto [sigma, u, v] = l0op::Svd(x, fullMatrices, computeUV, exe);
  ASSERT_NE(sigma, nullptr);
  ASSERT_NE(u, nullptr);
  ASSERT_NE(v, nullptr);
}

TEST_F(SvdTest, SvdTest_fullMatrices_false_SUCC) {
  auto x = CreateAclTensor({2, 3}, ACL_FLOAT);
  bool fullMatrices = false;
  bool computeUV= true;
  auto [sigma, u, v] = l0op::Svd(x, fullMatrices, computeUV, exe);
  ASSERT_NE(sigma, nullptr);
  ASSERT_NE(u, nullptr);
  ASSERT_NE(v, nullptr);
}

TEST_F(SvdTest, SvdTest_ComputeUV_false_SUCC) {
  auto x = CreateAclTensor({2, 3}, ACL_FLOAT);
  bool fullMatrices = false;
  bool computeUV= false;
  auto [sigma, u, v] = l0op::Svd(x, fullMatrices, computeUV, exe);
  ASSERT_NE(sigma, nullptr);
  ASSERT_NE(u, nullptr);
  ASSERT_NE(v, nullptr);
}
