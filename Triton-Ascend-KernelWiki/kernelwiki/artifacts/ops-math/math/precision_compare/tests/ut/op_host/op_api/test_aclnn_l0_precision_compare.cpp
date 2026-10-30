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

#include "opdev/make_op_executor.h"
#include "math/precision_compare/op_host/op_api/precision_compare.h"

const int64_t DATA_SIZE = 1024 * 1024;
class PrecisionCompareTest: public ::testing::Test {
public:
  PrecisionCompareTest() : exe(nullptr) {}
  aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dataType) {
    return aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           data);
  }
  
  void Clear() {}

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
  int64_t data[DATA_SIZE] = {0};
};

TEST_F(PrecisionCompareTest, DATA_TYPE_UINT32_SUCC) {
  auto errcode = CreateAclTensor({}, ACL_UINT32);
  auto benchMark = CreateAclTensor({4, 1, 6, 1}, ACL_FLOAT);
  auto realData = CreateAclTensor({4, 1, 6, 1}, ACL_FLOAT);
  auto out = l0op::PrecisionCompare(benchMark, realData, errcode, 0, exe);
  ASSERT_NE(out, nullptr);
}

TEST_F(PrecisionCompareTest, DATA_TYPE_BF16_SUCC) {
  auto errcode = CreateAclTensor({}, ACL_UINT32);
  auto benchMark = CreateAclTensor({4, 1, 6, 1}, ACL_BF16);
  auto realData = CreateAclTensor({4, 1, 6, 1}, ACL_BF16);
  auto out = l0op::PrecisionCompare(benchMark, realData, errcode, 1, exe);
  ASSERT_NE(out, nullptr);
}

TEST_F(PrecisionCompareTest, DATA_TYPE_EXCEPTION) {
  auto errcode = CreateAclTensor({}, ACL_UINT32);
  auto benchMark = CreateAclTensor({4, 1, 6, 1}, ACL_UINT32);
  auto realData = CreateAclTensor({4, 1, 6, 1}, ACL_FLOAT);
  auto out = l0op::PrecisionCompare(benchMark, realData, errcode, 0, exe);
  ASSERT_EQ(out, nullptr);
}