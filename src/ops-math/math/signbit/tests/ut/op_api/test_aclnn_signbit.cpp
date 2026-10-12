/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "math/signbit/op_api/aclnn_signbit.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

class l2_signbit_test : public testing::Test {
  protected:
  static void SetUpTestCase() { std::cout << "signbit_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "signbit_test TearDown" << std::endl; }
};

TEST_F(l2_signbit_test, case_support_float) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_signbit_test, case_support_float16) {
  auto self_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(0, 100);
  auto out_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE:only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_signbit_test, case_support_int64) {
  // use aicpu
  auto self_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(-5, 5);
  auto out_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE:only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_signbit_test, case_support_int32) {
  auto self_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-5, 5);
  auto out_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE:only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_signbit_test, case_support_int8) {
  auto self_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_INT8, ACL_FORMAT_NHWC).ValueRange(-5, 5);
  auto out_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE:only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_signbit_test, case_support_complex64) {
  auto self_tensor_desc = TensorDesc({6, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({6, 3}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_signbit_test, case_support_complex128) {
  auto self_tensor_desc = TensorDesc({6, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({6, 3}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_signbit_test, case_not_support_bool) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({10, 5}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(tensor_desc), OUTPUT(tensor_desc));

  // SAMPLE:only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空指针校验
TEST_F(l2_signbit_test, case_input_nullptr) {
  auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(nullptr), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_signbit_test, case_output_nullptr) {
  auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(tensor_desc), OUTPUT(nullptr));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空tensor
TEST_F(l2_signbit_test, case_empty_tensor) {
  auto self_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_signbit_test, case_support_hwcn) {
  auto self_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_HWCN).ValueRange(-5, 5);
  auto out_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_HWCN);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE:only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_signbit_test, case_support_ndhwc) {
  auto self_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_NDHWC).ValueRange(-5, 5);
  auto out_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_NDHWC);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE:only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_signbit_test, case_support_ncdhw) {
  auto self_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_NCDHW).ValueRange(-5, 5);
  auto out_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE:only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常维度校验
TEST_F(l2_signbit_test, case_invalid_dim) {
  auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_signbit_test, ascend910B2_input_output_bf16) {
  auto self_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BF16, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 4, 5, 6}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001,0.001);

  auto ut = OP_API_UT(aclnnSignbit, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

