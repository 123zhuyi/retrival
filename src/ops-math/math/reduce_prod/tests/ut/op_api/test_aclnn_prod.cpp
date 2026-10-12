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

class l2_prod_test : public testing::Test {
protected:
  static void SetUpTestCase() { std::cout << "prod_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "prod_test TearDown" << std::endl; }
};

TEST_F(l2_prod_test, case_null_tensor) {
  auto tensor_desc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_nullptr) {
  auto tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProd, INPUT(nullptr, ACL_FLOAT), OUTPUT(tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(nullptr));
  aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_prod_test, case_dim2_Float_Float_NCHW) {
  auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_error_dtype) {
  auto tensor_desc = TensorDesc({2, 2, 2}, ACL_UINT32, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_UINT32), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prod_test, case_error_shape) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prod_test, case_dim4_Int8_Int8_HWCN) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_HWCN);
  auto out = TensorDesc({}, ACL_INT8, ACL_FORMAT_HWCN); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_INT8), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_dim4_Float_HWCN_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_dim5_Uint8_Uint8_NDHWC) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_NDHWC);
  auto out = TensorDesc({}, ACL_UINT8, ACL_FORMAT_NDHWC); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_UINT8), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_dim5_Int16_Int32_NCDHW) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NCDHW);
  auto out = TensorDesc({}, ACL_INT32, ACL_FORMAT_NCDHW); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_INT32), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_dim5_Int32_Int32_NC1HWC0) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NC1HWC0);
  auto out = TensorDesc({}, ACL_INT32, ACL_FORMAT_NC1HWC0); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_INT32), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_dim8_Int64_Int64_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_INT64), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_dim6_Bool_Bool_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_BOOL), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_dim7_Complex64_Complex64_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 0, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_COMPLEX64, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_COMPLEX64), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prod_test, case_dim2_range) {
  auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, case_not_contiguous) {
  auto tensor_desc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, ascend950PR_89_case_dim6_Uint8_Bool_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_BOOL), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, ascend950PR_89_case_dim6_Int32_Bool_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_BOOL), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, ascend950PR_89_case_dim6_float16_fp32_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_prod_test, ascend950PR_89_case_dim6_float32_fp32_ND) {
  auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND); 

  auto ut = OP_API_UT(aclnnProd, INPUT(tensor_desc, ACL_FLOAT), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}