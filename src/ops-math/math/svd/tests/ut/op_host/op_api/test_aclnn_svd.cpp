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
#include <cstdio>
#include <vector>
#include <fstream>
#include "gtest/gtest.h"

#include "math/svd/op_host/op_api/aclnn_svd.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/inner/rts_interface.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_svd_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "tensor_svd_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "tensor_svd_test TearDown" << endl;
  }
};

// 正常调用流程
TEST_F(l2_svd_test, normal_input_success) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// input空指针
TEST_F(l2_svd_test, nullptr_input) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(nullptr, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// sigma空指针
TEST_F(l2_svd_test, nullptr_sigma) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(nullptr, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// u空指针
TEST_F(l2_svd_test, nullptr_u) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, nullptr, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// v空指针
TEST_F(l2_svd_test, nullptr_v) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, nullptr));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 类型不支持
TEST_F(l2_svd_test, type_unsupport) {
  auto inputDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 复数类型支持
TEST_F(l2_svd_test, type_complex) {
  auto inputDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// input 与 u 类型不一致
TEST_F(l2_svd_test, type_eq_input_u) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input 与 sigma 类型不一致
TEST_F(l2_svd_test, type_eq_input_sigma) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input 与 v 类型不一致
TEST_F(l2_svd_test, type_eq_input_v) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input shape 为空
TEST_F(l2_svd_test, shape_empty_input) {
  auto inputDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input dim > 8
TEST_F(l2_svd_test, shape_dim_over_8) {
  auto inputDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input dim < 2
TEST_F(l2_svd_test, shape_input_dim_less_than_2) {
  auto inputDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// fullMatrices=true时shape校验失败 - u
TEST_F(l2_svd_test, shape_fullMatrices_true_u) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// fullMatrices=true时shape校验失败 - sigma
TEST_F(l2_svd_test, shape_fullMatrices_true_sigma) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// fullMatrices=true时shape校验失败 - v
TEST_F(l2_svd_test, shape_fullMatrices_true_v) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = true;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// fullMatrices=fasle时shape校验失败 - u
TEST_F(l2_svd_test, shape_fullMatrices_false_u) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = false;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// fullMatrices=fasle时shape校验失败 - sigma
TEST_F(l2_svd_test, shape_fullMatrices_false_sigma) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = false;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// fullMatrices=fasle时shape校验失败 - v
TEST_F(l2_svd_test, shape_fullMatrices_false_v) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = false;
  bool computeUV = true;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ComputeUV=fasle时不再对U、V的shape进行校验
TEST_F(l2_svd_test, shape_fullMatrices_false_uv) {
  auto inputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  bool fullMatrices = false;
  bool computeUV = false;
  auto sigmaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto uDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnSvd, INPUT(inputDesc, fullMatrices, computeUV), OUTPUT(sigmaDesc, uDesc, vDesc));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}