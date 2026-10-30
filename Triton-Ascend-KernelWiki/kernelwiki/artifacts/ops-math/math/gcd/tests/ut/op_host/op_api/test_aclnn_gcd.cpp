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
#include "opdev/op_log.h"
#include "aclnn_gcd.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;


class l2_gcd_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Gcd Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Gcd Test TearDown" << std::endl; }
};

TEST_F(l2_gcd_test, aclnnGcd_01_int_nd) {
  auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, ascend910B2_aclnnGcd_02_int16_nd) {
  auto selfDesc = TensorDesc({2, 3, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 3, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto outDesc = TensorDesc({2, 3, 2}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, aclnnGcd_03_int_int16_hd) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto outDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, aclnnGcd_04_int16_int_hd) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_05_int_empty_tensor) {
  auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto otherDesc = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND);


  auto outDesc = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, aclnnGcd_06_int_nd_empty_tensor) {
  auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto otherDesc = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND);


  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_07_float_nd_empty_tensor) {
  auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto otherDesc = TensorDesc({1, 0, 1, 2}, ACL_INT16, ACL_FORMAT_ND);


  auto outDesc = TensorDesc({}, ACL_INT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_08_float_nd) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_09_float_nd_input_not_contiguous) {
  auto selfDesc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto otherDesc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);


  auto outDesc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, aclnnGcd_10_float_nd_inoutput_not_contiguous) {
  auto selfDesc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto otherDesc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);


  auto outDesc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, aclnnGcd_11_input_out_nullptr) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);


  auto ut_l = OP_API_UT(aclnnGcd, INPUT((aclTensor*)nullptr, tensor_desc), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_r = OP_API_UT(aclnnGcd, INPUT(tensor_desc, (aclTensor*)nullptr), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_r.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_o = OP_API_UT(aclnnGcd, INPUT(tensor_desc, tensor_desc), OUTPUT((aclTensor*)nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_o.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_gcd_test, aclnnGcd_12_aclnnGcd_input_error_shape) {
  auto selfDesc = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto otherDesc = TensorDesc({123, 8, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);


  auto outDesc = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_13_aclnnGcd_error_output_dtype) {
  auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto otherDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);


  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


TEST_F(l2_gcd_test, aclnnGcd_14_aclnnGcd_error_input_dtype) {
  auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto otherDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);


  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_15_input_dtype) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_16_error_input_dtype) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_17_input_dtype) {
  auto selfDesc = TensorDesc({2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_18_aclnnGcd_output_error_shape) {
  auto selfDesc = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto otherDesc = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);


  auto outDesc = TensorDesc({123, 8, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_19_aclnnGcd_output_error_shape) {
  auto selfDesc = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto otherDesc = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);


  auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_20_aclnnGcd_input_error_shape_len) {
  auto tensorDesc9 = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto tensorDesc1 = TensorDesc({7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto tensorDesc2 = TensorDesc({7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);


  auto ut_grad = OP_API_UT(aclnnGcd, INPUT(tensorDesc9, tensorDesc1),
                           OUTPUT(tensorDesc2));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_grad.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  auto ut_self = OP_API_UT(aclnnGcd, INPUT(tensorDesc1, tensorDesc9),
                           OUTPUT(tensorDesc2));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_21_input_dtype) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_22_input_dtype) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, aclnnGcd_23_input_dtype) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// int64正常场景用例
TEST_F(l2_gcd_test, ascend910B2_aclnnGcd_24_int64_nd) {
  auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, ascend910B2_aclnnGcd_25_int64_nd) {
  auto selfDesc = TensorDesc({2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto outDesc = TensorDesc({2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, ascend910B2_aclnnGcd_26_int64_nd) {
  auto selfDesc = TensorDesc({2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto outDesc = TensorDesc({2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_gcd_test, ascend910B2_aclnnGcd_27_int64_int16_hd) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto outDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// int64异常场景用例
TEST_F(l2_gcd_test, aclnnGcd_28_int_int64_hd) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto otherDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);


  auto outDesc = TensorDesc({}, ACL_INT64, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
