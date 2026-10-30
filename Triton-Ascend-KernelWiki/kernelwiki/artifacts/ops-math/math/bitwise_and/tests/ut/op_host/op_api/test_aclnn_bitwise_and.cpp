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

#include "aclnn_bitwiseand.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_bitwise_and_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "bitwise_and_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "bitwise_and_test TearDown" << endl; }
};

TEST_F(l2_bitwise_and_test, case_int16) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_bitwise_and_test, case_int32) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_bitwise_and_test, case_int64) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_bitwise_and_test, case_int8) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_bitwise_and_test, case_uint8) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_UINT8, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_UINT8, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_bitwise_and_test, case_uint16) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_UINT16, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_UINT16, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_bitwise_and_test, case_bool) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_bitwise_and_test, case_suppport_format) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCHW);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCHW);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();

  self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC);
  other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC);
  out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut1 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut1.TestPrecision();

  self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_HWCN);
  other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_HWCN);
  out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut2 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut2.TestPrecision();

  self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NDHWC);
  other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NDHWC);
  out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut3 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut3.TestPrecision();

  self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCDHW);
  other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCDHW);
  out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut4 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut4.TestPrecision();

}

TEST_F(l2_bitwise_and_test, case_nullptr) {
  auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空tensor
TEST_F(l2_bitwise_and_test, case_empty) {
  auto self_tensor_desc = TensorDesc({7, 0, 6}, ACL_INT32, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({7, 1, 6}, ACL_INT32, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_test, case_errtype) {
  auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_bitwise_and_test, case_nullptr2) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);

  auto ut_2 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(nullptr, tensor_desc), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_3 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(tensor_desc, nullptr), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_4 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(tensor_desc, tensor_desc), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_bitwise_and_test, case_dtype_uint32) {
  //uint32
  auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(tensor_desc, tensor_desc), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  //uint64
  tensor_desc = TensorDesc({10, 5}, ACL_UINT64, ACL_FORMAT_ND);

  auto ut_3 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(tensor_desc, tensor_desc), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  //float
  tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut_4 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(tensor_desc, tensor_desc), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut_4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

// CheckShape
TEST_F(l2_bitwise_and_test, case_shape) {
  auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_INT32, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_INT32, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  self_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_INT32, ACL_FORMAT_ND);
  other_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_INT32, ACL_FORMAT_ND);
  out_tensor_desc = TensorDesc({10, 5, 10, 5}, ACL_INT32, ACL_FORMAT_ND);

  auto ut_2 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat
TEST_F(l2_bitwise_and_test, case_format) {
  auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_INT32, ACL_FORMAT_NHWC);
  auto other_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_INT32, ACL_FORMAT_NCHW);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut_1 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  auto tensor_desc = TensorDesc({10, 5, 2, 10, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

  auto ut_2 = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(tensor_desc, tensor_desc), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeOnMilan
TEST_F(l2_bitwise_and_test, ascend910B2_case_int64) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
  auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnBitwiseAndTensorOut, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

