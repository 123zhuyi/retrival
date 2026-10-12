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

#include "math/digamma/op_host/op_api/aclnn_digamma.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_digamma_test : public testing::Test {
  protected:
    static void SetUpTestCase() { cout << "digamma_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "digamma_test TearDown" << endl; }
};

// *** tensor dtype test ***
// test aicore type: FLOAT/FLOAT32
TEST_F(l2_digamma_test, case_digamma_for_float_type) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: FLOAT16
TEST_F(l2_digamma_test, case_digamma_for_float16_type) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicpu type: DOUBLE/FLOAT64, in fact tf mode
TEST_F(l2_digamma_test, case_digamma_for_double_type) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_digamma_test, case_invalid_input_type) {
  // complex64
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // INT32
  self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  out_tensor_desc = TensorDesc(self_tensor_desc);

  ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  workspace_size = 0;
  aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // INT8
  self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  out_tensor_desc = TensorDesc(self_tensor_desc);

  ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  workspace_size = 0;
  aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // INT16
  self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  out_tensor_desc = TensorDesc(self_tensor_desc);

  ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  workspace_size = 0;
  aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // INT64
  self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
  out_tensor_desc = TensorDesc(self_tensor_desc);

  ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  workspace_size = 0;
  aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** tensor format test ***
// test format NCHW
TEST_F(l2_digamma_test, case_digamma_for_nchw_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// test format NHWC
TEST_F(l2_digamma_test, case_digamma_for_nhwc_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// test format HWCN
TEST_F(l2_digamma_test, case_digamma_for_hwcn_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_DOUBLE, ACL_FORMAT_HWCN).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// test invalid format
TEST_F(l2_digamma_test, case_invalid_format_with_nc1hwc0) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// *** tensor rank range ***
// empty tensor
TEST_F(l2_digamma_test, case_aicore_digamma_for_empty_tensor) {
  auto self_tensor_desc = TensorDesc({0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test normal rank with four dim
TEST_F(l2_digamma_test, case_digamma_for_normal_rank) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// test abnormal rank with right boundary dim, nine
TEST_F(l2_digamma_test, case_digamma_for_abnormal_right_boundary_rank) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** tensor relationship constraint test ***
// test invalid input with diff dtype
TEST_F(l2_digamma_test, case_invalid_input_with_diff_dtype) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test invalid input with diff shape
TEST_F(l2_digamma_test, case_invalid_input_with_diff_shape) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test invalid input with diff format
TEST_F(l2_digamma_test, case_invalid_input_with_diff_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnDigamma, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}