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

#include "../../../op_api/aclnn_addr.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_addr_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "addr_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "addr_test TearDown" << endl;
  }
};

// *** tensor dtype test ***
// test dtype: FLOAT16
TEST_F(l2_addr_test, case_addr_for_float16_type) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// test dtype: BFLOAT16
TEST_F(l2_addr_test, ascend910B2_addr_dtype_bfloat16) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test dtype: BFLOAT16 91095
TEST_F(l2_addr_test, ascend950_addr_dtype_bfloat16) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test dtype: FLOAT/FLOAT32
TEST_F(l2_addr_test, case_addr_for_float_type) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto beta_scalar_desc = ScalarDesc(1.5f);
  auto alpha_scalar_desc = ScalarDesc(1.1f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// test dtype: INT8
TEST_F(l2_addr_test, case_addr_for_int8_type) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(static_cast<int8_t>(5));
  auto alpha_scalar_desc = ScalarDesc(static_cast<int8_t>(-5));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// test dtype: INT32
TEST_F(l2_addr_test, case_addr_for_int32_type) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(static_cast<int32_t>(-10));
  auto alpha_scalar_desc = ScalarDesc(static_cast<int32_t>(2));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// test dtype: INT64
TEST_F(l2_addr_test, case_addr_for_int64_type) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(static_cast<int64_t>(3));
  auto alpha_scalar_desc = ScalarDesc(static_cast<int64_t>(1));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// test dtype: UINT8
TEST_F(l2_addr_test, case_addr_for_uint8_type) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(static_cast<uint8_t>(2));
  auto alpha_scalar_desc = ScalarDesc(static_cast<uint8_t>(2));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// test dtype: BOOL
TEST_F(l2_addr_test, case_addr_for_bool_type) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(false, true);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(false, true);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(false, true);
  auto beta_scalar_desc = ScalarDesc(static_cast<bool>(true));
  auto alpha_scalar_desc = ScalarDesc(static_cast<bool>(true));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// test invalid input type
TEST_F(l2_addr_test, case_addr_for_invalid_complex_type) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input tensor is not bool tensor when beta/alpha's dtype is boolean
TEST_F(l2_addr_test, case_addr_for_bool_scalar_without_bool_input) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(false, true);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(false, true);
  auto beta_scalar_desc = ScalarDesc(static_cast<bool>(true));
  auto alpha_scalar_desc = ScalarDesc(static_cast<bool>(true));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// all input tensors have integal type, but beta/alpha has floating type
TEST_F(l2_addr_test, case_addr_for_floating_scalar_under_int_input) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** tensor rank range ***
// vec1 is a empty tensor
TEST_F(l2_addr_test, case_addr_for_left_vec_is_empty_tensor) {
  auto self_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// vec2 is a empty tensor
TEST_F(l2_addr_test, case_addr_for_right_vec_is_empty_tensor) {
  auto self_tensor_desc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// vec is not a 1D tensor
TEST_F(l2_addr_test, case_addr_for_vec_isnot_1d_tensor) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self dims is over 2
TEST_F(l2_addr_test, case_addr_for_self_dims_over_two) {
  auto self_tensor_desc = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self can not broadcast to outer of vec1 and vec2
TEST_F(l2_addr_test, case_addr_for_unable_broadcast_within_self_vec12) {
  auto self_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** tensor relationship constraint test ***
// vecs have different dtype, test dtype cast
TEST_F(l2_addr_test, case_addr_for_vec_dtype_cast) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(static_cast<float>(100.200000));
  auto alpha_scalar_desc = ScalarDesc(static_cast<float>(200.500000));
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// *** nullptr test ***
// test nullptr input
TEST_F(l2_addr_test, case_addr_for_nullptr_input) {
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT((aclTensor*)nullptr, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr vec
TEST_F(l2_addr_test, case_addr_for_nullptr_vec) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, (aclTensor*)nullptr, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr output
TEST_F(l2_addr_test, case_addr_for_nullptr_output) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, (aclTensor*)nullptr, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT((aclTensor*)nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr scalar, handle with default value
TEST_F(l2_addr_test, case_addr_for_nullptr_scalar) {
  auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, (aclScalar*)nullptr, (aclScalar*)nullptr),
                      OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}