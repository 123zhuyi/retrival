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

#include "aclnn_scale.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2ScaleTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2ScaleTest SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "l2ScaleTest TearDown" << endl;
  }
};

TEST_F(l2ScaleTest, ascend910B2_normal_1) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorBias = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t axis = 1;
  const int64_t numAxes = 1;
  const bool scaleFromBlob = true;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, tensorBias, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ScaleTest, ascend910B2_normal_2) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorBias = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t axis = 1;
  int64_t numAxes = 1;
  bool scaleFromBlob = false;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, tensorBias, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ScaleTest, ascend910B2_bias_nullptr) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t axis = 1;
  int64_t numAxes = 1;
  bool scaleFromBlob = false;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, (aclTensor *)nullptr, axis, numAxes, scaleFromBlob),
                OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ScaleTest, ascend910B2_x_scale_dtype_not_same) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t axis = 1;
  int64_t numAxes = 1;
  bool scaleFromBlob = false;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, nullptr, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ScaleTest, ascend910B2_x_y_dtype_not_same) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  int64_t axis = 1;
  int64_t numAxes = 1;
  bool scaleFromBlob = false;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, nullptr, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ScaleTest, ascend910B2_scale_0) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t axis = 1;
  const int64_t numAxes = 1;
  const bool scaleFromBlob = false;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, nullptr, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ScaleTest, ascend910B2_scale_2) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t axis = 0;
  int64_t numAxes = 2;
  bool scaleFromBlob = true;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, nullptr, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ScaleTest, ascend910B2_axis_invalid) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t axis = -3;
  int64_t numAxes = 2;
  bool scaleFromBlob = true;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, nullptr, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ScaleTest, ascend910B2_num_axes_invalid) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t axis = -1;
  int64_t numAxes = -2;
  bool scaleFromBlob = true;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, nullptr, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ScaleTest, ascend910B2_scalar_true) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorBias = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t axis = 0;
  int64_t numAxes = 0;
  bool scaleFromBlob = true;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, tensorBias, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ScaleTest, ascend910B2_scalar_false) {
  auto tensorX = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2)
                     .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto tensorScale = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorBias = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorOut = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t axis = 0;
  int64_t numAxes = 0;
  bool scaleFromBlob = false;

  auto ut =
      OP_API_UT(aclnnScale, INPUT(tensorX, tensorScale, tensorBias, axis, numAxes, scaleFromBlob), OUTPUT(tensorOut));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}