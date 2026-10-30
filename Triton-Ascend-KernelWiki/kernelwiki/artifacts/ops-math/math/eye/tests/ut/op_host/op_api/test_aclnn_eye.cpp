/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "aclnn_eye.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;


class eye_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "eye_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "eye_test TearDown" << endl; }
};

TEST_F(eye_test, case_norm_float32) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(eye_test, case_norm_float16) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(eye_test, case_norm_int32) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(eye_test, case_norm_int8) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT8, ACL_FORMAT_ND)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(eye_test, case_norm_uint8) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_UINT8, ACL_FORMAT_ND)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(eye_test, case_norm_int16) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// TEST_F(eye_test, case_nullptr_out) {
//   int64_t n = 4;
//   int64_t m = 4;
//   auto out_tensor_desc = nullptr;

//   auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

TEST_F(eye_test, case_dtype_invalid_out) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(eye_test, case_format_NCHW) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(eye_test, case_format_internal) {
  int64_t n = 2;
  int64_t m = 2;
  auto out_tensor_desc = TensorDesc({2,2}, ACL_FLOAT, ACL_FORMAT_NC1HWC0)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(eye_test, case_empty_tensor) {
  int64_t n = 0;
  int64_t m = 0;
  auto out_tensor_desc = TensorDesc({0,0}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(eye_test, case_dim_invalid) {
  int64_t n = 2;
  int64_t m = 2;
  auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(eye_test, case_format_abnormal) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_UNDEFINED)
                         .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(eye_test, case_dtype_abnormal) {
  int64_t n = 4;
  int64_t m = 4;
  auto out_tensor_desc = TensorDesc({4, 4}, ACL_DT_UNDEFINED, ACL_FORMAT_NHWC)
                           .ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnEye, INPUT(n, m), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}