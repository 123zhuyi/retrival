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

#include "../../../op_api/aclnn_prod.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_prod_dim_test : public testing::Test {
protected:
  static void SetUpTestCase() { std::cout << "prod_dim_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "prod_dim_test TearDown" << std::endl; }
};

TEST_F(l2_prod_dim_test, case_null_tensor) {
  auto tensor_desc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 1;
  auto out = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim0) {
  auto tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_nullptr) {
  auto tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(nullptr, dim, true, ACL_FLOAT), OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut1 = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(nullptr));
  aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_prod_dim_test, case_dim1_Float_Float_ND) {
  auto tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_error_dtype) {
  auto tensor_desc = TensorDesc({2, 2, 2}, ACL_UINT32, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out = TensorDesc({2, 2}, ACL_UINT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, false, ACL_UINT32), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prod_dim_test, case_diff_dtype) {
  auto tensor_desc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, false, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prod_dim_test, case_error_dim) {
  auto tensor_desc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 5;
  auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, false, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prod_dim_test, case_error_input_shape) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, false, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prod_dim_test, case_error_out_shape) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, false, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prod_dim_test, case_dim4_Int8_Int8_HWCN) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_HWCN);
  int64_t dim = 3;
  auto out = TensorDesc({2, 2, 2, 1}, ACL_INT8, ACL_FORMAT_HWCN);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_INT8), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim4_Float_HWCN_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);
  int64_t dim = 3;
  auto out = TensorDesc({2, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim5_Uint8_Uint8_NDHWC) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  int64_t dim = 4;
  auto out = TensorDesc({2, 2, 2, 2, 1}, ACL_UINT8, ACL_FORMAT_NDHWC);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_UINT8), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim5_Int16_Int32_NCDHW) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NCDHW);
  int64_t dim = 4;
  auto out = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NCDHW);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, false, ACL_INT32), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim5_Int32_Int32_NC1HWC0) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NC1HWC0);
  int64_t dim = 3;
  auto out = TensorDesc({2, 2, 2, 1, 2}, ACL_INT32, ACL_FORMAT_NC1HWC0);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_INT32), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim8_Int64_Int64_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
  int64_t dim = 7;
  auto out = TensorDesc({2, 2, 2, 2, 2, 2, 2, 1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_INT64), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim6_Bool_Bool_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);
  int64_t dim = 5;
  auto out = TensorDesc({2, 2, 2, 2, 2, 1}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_BOOL), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim2_range) {
  auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t dim = 1;
  auto out = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_not_contiguous) {
  auto tensor_desc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
  int64_t dim = 1;
  auto out = TensorDesc({3, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_dim2_neg_dim) {
  auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t dim = -1;
  auto out = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_fp16) {
  auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t dim = -1;
  auto out = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, case_uint8) {
  auto tensor_desc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t dim = -1;
  auto out = TensorDesc({2, 1}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_BOOL), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_dim_test, ascend910B1_case_fp32) {
  auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t dim = -1;
  auto out = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnProdDim, INPUT(tensor_desc, dim, true, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}