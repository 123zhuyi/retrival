/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_polar.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_polar_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "polar_test  SetUp" << endl; }

  static void TearDownTestCase() { cout << "polar_test TearDown" << endl; }
};

TEST_F(l2_polar_test, case_fp32) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto outDesc = TensorDesc({1, 2}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_polar_test, case_mismatch_dtype) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_empty) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT((aclTensor*)nullptr, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

}

TEST_F(l2_polar_test, case_empty2) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, (aclTensor*)nullptr), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

}

TEST_F(l2_polar_test, case_unsupport1) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_max_dim1) {
  // input
  auto inputDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 5, 6}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_max_dim2) {
  // input
  auto inputDesc = TensorDesc({1, 1}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_broadcast_failed) {
  // input
  auto inputDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 4}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}


TEST_F(l2_polar_test, case_max_dim3) {
  // input
  auto inputDesc = TensorDesc({1, 1}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_dim_not_match1) {
  // input
  auto inputDesc = TensorDesc({1, 2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_dim_not_match2) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_dim_not_match3) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_empty_input_1) {
  // input
  auto inputDesc = TensorDesc({0, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_empty_input_2) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({0, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_polar_test, case_empty_input_3) {
  // input
  auto inputDesc = TensorDesc({1, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto angleDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,1);
  auto outDesc = TensorDesc({0, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut = OP_API_UT(aclnnPolar, INPUT(inputDesc, angleDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}