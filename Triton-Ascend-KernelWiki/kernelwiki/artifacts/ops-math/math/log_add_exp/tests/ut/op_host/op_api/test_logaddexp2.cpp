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

#include "../../../../op_host/op_api/aclnn_logaddexp2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_logaddexp2_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "logaddexp2_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "logaddexp2_test TearDown" << endl;
  }
};

TEST_F(l2_logaddexp2_test, case_nullptr_input) {
  auto tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_logaddexp2_test, case_nullptr_output) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT((aclTensor*)nullptr));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_logaddexp2_test, case_dtype_notsupport) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logaddexp2_test, case_dtype_input_int_out_fp) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_dtype_input_bool_out_fp) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_dtype_fp32) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_dtype_fp16) {
  auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_dtype_double) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_format_HWCN) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_format_NDHWC) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_format_NCDHW) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto other_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_promoteType_ok) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_promoteType_nok) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logaddexp2_test, case_invalid_shape) {
  auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logaddexp2_test, case_broadcast_ok) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_broadcast_nok) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({3, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logaddexp2_test, case_dim_1) {
  auto self_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_dim_3) {
  auto self_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_dim_4) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_dim_8) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_dim_9) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logaddexp2_test, case_empty_empty_empty) {
  auto self_tensor_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({0, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({0, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_empty_noempty_empty) {
  auto self_tensor_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_noempty_empty_empty) {
  auto self_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({0, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({0, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_logaddexp2_test, case_noempty_noempty_empty) {
  auto self_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({0, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logaddexp2_test, case_empty_empty_noempty) {
  auto self_tensor_desc = TensorDesc({0, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logaddexp2_test, case_noempty_empty_noempty) {
  auto self_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logaddexp2_test, case_empty_noempty_noempty) {
  auto self_tensor_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto other_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
