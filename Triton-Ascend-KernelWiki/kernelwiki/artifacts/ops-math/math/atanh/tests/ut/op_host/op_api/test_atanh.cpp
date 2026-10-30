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

#include "../../../../op_host/op_api/aclnn_atanh.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_atanh_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_atanh_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "l2_atanh_test TearDown" << endl;
  }
};

TEST_F(l2_atanh_test, atanh_int_float) {
  auto self_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atanh_test, atanh_float_float16) {
  auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atanh_test, ascend910B2_atanh_bfloat16_bfloat16) {
  auto self_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atanh_test, atanh_float_int) {
  auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_atanh_test, atanh_different_shape) {
  auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_atanh_test, atanh_nullptr) {
  auto ut = OP_API_UT(aclnnAtanh, INPUT((aclTensor*)nullptr), OUTPUT((aclTensor*)nullptr));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_atanh_test, atanh_out_nullptr) {
  auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(nullptr));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_atanh_test, atanh_precision) {
  auto self_tensor_desc = TensorDesc({3, 6, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({3, 6, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atanh_test, atanh_empty_tensor) {
  auto self_tensor_desc = TensorDesc({13, 0, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({13, 0, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atanh_test, atanh_non_contiguous) {
  auto self_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atanh_test, atanh_lessDim) {
  auto self_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atanh_test, atanh_bigDim) {
  auto self_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnAtanh, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_atanh_test, atanh_inplace) {
  auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto ut = OP_API_UT(aclnnInplaceAtanh, INPUT(self_tensor_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
