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

#include "reduce_std_with_mean/op_host/op_api/aclnn_batch_norm_stats.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_batch_norm_stats_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_batch_norm_stats_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "l2_batch_norm_stats_test TearDown" << endl;
  }
};

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_dtype_float) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  ut.TestPrecision();
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_dtype_float16) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  ut.TestPrecision();
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_dtype_int8) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_dtype_int32) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_dtype_uint8) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_dtype_int16) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  uint64_t workspace_size = 0;
  auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_all_format) {
  vector<aclFormat> format_list{ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW,  ACL_FORMAT_NHWC, ACL_FORMAT_ND,
                                ACL_FORMAT_HWCN,    ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
  for (auto format : format_list) {
    cout << "+++++++++++++++++++++++ start to test format " << format << endl;
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, format).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto mean = TensorDesc({3}, ACL_FLOAT, format);
    auto invstd = TensorDesc({3}, ACL_FLOAT, format);
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (format == ACL_FORMAT_NC1HWC0) {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
      EXPECT_EQ(aclRet, ACL_SUCCESS);
      ut.TestPrecision();
    }
  }
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_nullptr) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  auto ut_1 = OP_API_UT(aclnnBatchNormStats, INPUT(nullptr, eps), OUTPUT(mean, invstd));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_2 = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(nullptr, invstd));
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_3 = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, nullptr));
  aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_empty_tensor) {
  auto tensor_desc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  auto ut_1 = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut_2 = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_non_contiguous) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2}).ValueRange(-1, 1);
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_norm_stats_test, batch_norm_stats_invalid_dim) {
  auto tensor_desc = TensorDesc({2, 3, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  auto ut_1 = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut_2 = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_norm_stats_test, ascend950_batch_norm_stats_dtype_float16) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto invstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  double eps = 1e-5;

  auto ut = OP_API_UT(aclnnBatchNormStats, INPUT(tensor_desc, eps), OUTPUT(mean, invstd));
  uint64_t workspace_size = 0;
  ut.TestGetWorkspaceSize(&workspace_size);
}
