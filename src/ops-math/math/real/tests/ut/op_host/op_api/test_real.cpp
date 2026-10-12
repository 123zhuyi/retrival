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

#include "aclnn_real.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_real_test : public testing::Test {
  protected:
    static void SetUpTestCase() { cout << "real_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "real_test TearDown" << endl; }
};

// *** tensor dtype test ***
// test aicore type: ACL_COMPLEX64
TEST_F(l2_real_test, case_aicore_real_for_complex64_type) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// test aicore type: ACL_COMPLEX32
TEST_F(l2_real_test, case_aicore_real_for_complex32_type) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX32, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  if ((GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) &&
      (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93)) {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  } else {
      EXPECT_EQ(aclRet, ACL_SUCCESS);
      ut.TestPrecision();
  }
}

// test aicore type: ACL_COMPLEX128
TEST_F(l2_real_test, case_aicore_real_for_complex128_type) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// test aicore type: FLOAT/FLOAT32
TEST_F(l2_real_test, case_aicore_real_for_float_type) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  if ((GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) &&
      (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93)) {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  } else {
      EXPECT_EQ(aclRet, ACL_SUCCESS);
      ut.TestPrecision();
  }
}

// test aicore type: FLOAT16
TEST_F(l2_real_test, case_aicore_real_for_float16_type) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  if ((GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) &&
      (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93)) {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  } else {
      EXPECT_EQ(aclRet, ACL_SUCCESS);
      ut.TestPrecision();
  }
}

// test invalid input type
TEST_F(l2_real_test, case_invalid_input_type) {
  // undefined
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** tensor format test ***
// test format NCHW
TEST_F(l2_real_test, case_real_for_nchw_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_NCHW);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// test format NHWC
TEST_F(l2_real_test, case_real_for_nhwc_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_NHWC);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// test format HWCN
TEST_F(l2_real_test, case_real_for_hwcn_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_HWCN);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_HWCN);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// test format NDHWC
TEST_F(l2_real_test, case_real_for_ndhwc_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_NDHWC);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// test format NCDHW
TEST_F(l2_real_test, case_real_for_ncdhw_format) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_NCDHW);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// test invalid format
TEST_F(l2_real_test, case_invalid_format_with_nc1hwc0) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_NC1HWC0);
  auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// *** tensor rank range ***
// empty tensor
TEST_F(l2_real_test, case_aicore_real_for_empty_tensor) {
  auto self_tensor_desc = TensorDesc({0, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({0, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test normal rank with four dim
TEST_F(l2_real_test, case_real_for_normal_rank) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// test abnormal rank with right boundary dim, nine
TEST_F(l2_real_test, case_real_for_abnormal_right_boundary_rank) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test abnormal rank with right boundary dim, nine
TEST_F(l2_real_test, case_real_for_abnormal_dtype) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test invalid input with diff shape
TEST_F(l2_real_test, case_invalid_input_with_diff_shape) {
  auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** nullptr test ***
// test nullptr input
TEST_F(l2_real_test, case_anullptr_input) {
  auto out_tensor_desc = TensorDesc({2, 2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT((aclTensor*)nullptr), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr output
TEST_F(l2_real_test, case_anullptr_output) {
  auto self_tensor_desc = TensorDesc({2, 2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT((aclTensor*)nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// *** contiguous ***
// test continuity
TEST_F(l2_real_test, case_real_for_continuity) {
  auto self_tensor_desc = TensorDesc({5, 4}, ACL_COMPLEX64, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}