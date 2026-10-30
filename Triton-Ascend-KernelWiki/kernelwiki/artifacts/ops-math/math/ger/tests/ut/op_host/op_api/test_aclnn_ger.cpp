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
 * \file test_ger.cpp
 * \brief
 */
 
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "aclnn_ger.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;


class ger_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "ger_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "ger_test TearDown" << endl; }
};

TEST_F(ger_test, case_test_001_float32) {
  auto self_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(ger_test, case_test_002_float16) {
  auto self_desc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(ger_test, case_test_004_int8) {
  auto self_desc = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_INT8, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(ger_test, case_test_005_uint8) {
  auto self_desc = TensorDesc({4}, ACL_UINT8, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_UINT8, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(ger_test, case_test_007_int32) {
  auto self_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(ger_test, case_test_008_int64) {
  auto self_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(ger_test, case_test_009_bool) {
  auto self_desc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(ger_test, case_test_010_complex64) {
  auto self_desc = TensorDesc({4}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

TEST_F(ger_test, case_test_012_self_nullptr) {
  auto vec2_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(nullptr, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(ger_test, case_test_013_out_nullptr) {
  auto self_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(ger_test, case_test_014_self_fp16_vec_fp32) {
  auto self_desc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(ger_test, case_test_015_error_self_size) {
  auto self_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ger_test, case_test_016_error_out_shape) {
  auto self_desc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ger_test, case_test_017_error_vec_shape) {
  auto self_desc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ger_test, case_test_018_self_empty) {
  auto self_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto vec2_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGer, INPUT(self_desc, vec2_desc), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

