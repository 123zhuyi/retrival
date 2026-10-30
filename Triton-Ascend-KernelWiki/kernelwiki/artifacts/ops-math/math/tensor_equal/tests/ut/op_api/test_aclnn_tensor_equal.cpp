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
 * \file tensor_equal.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "math/tensor_equal/op_api/aclnn_equal.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2_tensor_equal_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "tensor_equal_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "tensor_equal_test TearDown" << endl; }
};

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_float16) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 正常场景 true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal01) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 正常场景 FORMAT不相等 true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal03) {
  auto tensor_self = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11, 3, 6, 3, 4, 9, 6, 7, 11, 3, 6});

  auto tensor_other = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11, 3, 6, 3, 4, 9, 6, 7, 11, 3, 6});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{false});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 正常场景 self空   other空 -> true
TEST_F(l2_tensor_equal_test, test_tensor_equal_normal06) {

  auto tensor_self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2,1,3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{1,2,4,3,2,1});

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_tensor_equal_test, test_tensor_equal_normal_double) {
  auto tensor_self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND)
                          .Value(vector<bool>{true});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  // SAMPLE: precision simulate
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// out数据类型不是bool的场景
TEST_F(l2_tensor_equal_test, test_tensor_equal_out_dtype_not_bool) {
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);

  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1});

  auto ut = OP_API_UT(aclnnEqual, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 5UL);
}