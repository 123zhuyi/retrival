/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include "math/logdet/op_host/op_api/aclnn_logdet.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_logdet_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "logdet_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "logdet_test TearDown" << endl; }
};

// self == nullptr
TEST_F(l2_logdet_test, ascend910B2_case_nullptr_self) {
  auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(nullptr), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// logOut == nullptr
TEST_F(l2_logdet_test, ascend910B2_case_nullptr_logout) {
  auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常数据类型 self
TEST_F(l2_logdet_test, ascend910B2_case_fp16_self) {
  auto self = TensorDesc({3, 5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常数据类型 logOut
TEST_F(l2_logdet_test, ascend910B2_case_fp16_logout) {
  auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常数据类型 self是复数但logOut是实数
TEST_F(l2_logdet_test, ascend910B2_case_dtype_can_not_promote) {
  auto self = TensorDesc({3, 5, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据类型覆盖 float
TEST_F(l2_logdet_test, ascend910B2_case_float) {
  auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 数据类型覆盖 double
TEST_F(l2_logdet_test, ascend910B2_case_double) {
  auto self = TensorDesc({3, 5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 数据类型覆盖 complex64
TEST_F(l2_logdet_test, ascend910B2_case_complex64) {
  auto self = TensorDesc({3, 5, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 数据类型覆盖 complex128
TEST_F(l2_logdet_test, ascend910B2_case_complex128) {
  auto self = TensorDesc({3, 5, 5}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 8维以上
TEST_F(l2_logdet_test, ascend910B2_case_dim_10) {
  auto self = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3, 5, 5}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// self不是一系列方阵
TEST_F(l2_logdet_test, ascend910B2_case_self_not_equal_noise_1) {
  auto self = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logOut不是self去掉最后两维
TEST_F(l2_logdet_test, ascend910B2_case_sign_out_shape_wrong) {
  auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// format覆盖
TEST_F(l2_logdet_test, ascend910B2_case_format) {
  vector<aclFormat> ValidList = {
      ACL_FORMAT_UNDEFINED, ACL_FORMAT_NCHW,        ACL_FORMAT_NHWC, ACL_FORMAT_ND,    ACL_FORMAT_NC1HWC0,
      ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NC1HWC0_C04, ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_FRACTAL_NZ,
      ACL_FORMAT_NCDHW,     ACL_FORMAT_NDC1HWC0,    ACL_FRACTAL_Z_3D};

  int length = ValidList.size();
  for (int i = 0; i < length; i++) {
    auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  }
}

// 输入输出dtype不一致但可以cast
TEST_F(l2_logdet_test, ascend910B2_case_dtype_inconsistent) {
  auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 输入输出format不一致
TEST_F(l2_logdet_test, ascend910B2_case_format_inconsistent) {
  auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空tensor
TEST_F(l2_logdet_test, ascend910B2_case_empty_tensor_training) {
  auto self = TensorDesc({0, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto logOut = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 非连续tensor
TEST_F(l2_logdet_test, ascend910B2_case_not_contiguous_1) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {5, 5}).ValueRange(0, 2);
  auto logOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

  auto ut = OP_API_UT(aclnnLogdet, INPUT(self), OUTPUT(logOut));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}