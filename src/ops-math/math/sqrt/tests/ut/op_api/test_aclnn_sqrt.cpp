/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_sqrt.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_sqrt_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_Sqrt_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "l2_Sqrt_test TearDown" << std::endl; }
};

// self的数据类型不在支持范围内
TEST_F(l2_sqrt_test, l2_Sqrt_test_001) {
  auto selfDesc = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，COMPLEX128
// TEST_F(l2_sqrt_test, l2_Sqrt_test_002) {
//   auto selfDesc = TensorDesc({2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);
//   auto outDesc = TensorDesc({2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.001, 0.001);

//   auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

//   uint64_t workspaceSize = 0;
//   aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // ut.TestPrecision();
// }

TEST_F(l2_sqrt_test, l2_Sqrt_test_003) {
  auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                        .ValueRange(0, 2)
                        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 461, 16});;
  auto out_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(self_desc), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 空tensor
TEST_F(l2_sqrt_test, l2_Sqrt_test_004) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float
TEST_F(l2_sqrt_test, l2_Sqrt_test_005) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // ut.TestPrecision();
}


// 正常路径，COMPLEX64
// TEST_F(l2_sqrt_test, l2_Sqrt_test_006) {
//   auto selfDesc = TensorDesc({2, 4}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);
//   auto outDesc = TensorDesc({2, 4}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);

//   auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

//   uint64_t workspaceSize = 0;
//   aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // ut.TestPrecision();
// }

// 正常路径，float
TEST_F(l2_sqrt_test, l2_Sqrt_test_007) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(0, 2);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // ut.TestPrecision();
}

// 正常路径，float
TEST_F(l2_sqrt_test, l2_Sqrt_test_008) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(0, 2);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // ut.TestPrecision();
}

// 正常路径，float64
// TEST_F(l2_sqrt_test, l2_Sqrt_test_009) {
//   auto selfDesc = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
//   auto outDesc = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);

//   auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

//   uint64_t workspaceSize = 0;
//   aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // ut.TestPrecision();
// }

// 正常路径，float64
// TEST_F(l2_sqrt_test, l2_Sqrt_test_010) {
//   auto selfDesc = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
//   auto outDesc = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);

//   auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

//   uint64_t workspaceSize = 0;
//   aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

//   ut.TestPrecision();
// }

// 正常路径，int64
// TEST_F(l2_sqrt_test, l2_Sqrt_test_011) {
//   auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
//   auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

//   auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

//   uint64_t workspaceSize = 0;
//   aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  // ut.TestPrecision();
// }

// shape不一致
TEST_F(l2_sqrt_test, l2_Sqrt_test_012) {
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({4, 7}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// shape不一致
TEST_F(l2_sqrt_test, l2_Sqrt_test_013) {
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({4, 7}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(nullptr), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 维度过大
TEST_F(l2_sqrt_test, l2_Sqrt_test_014 ) {
  auto selfDesc = TensorDesc({1,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSqrt, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self的数据类型不在支持范围内
TEST_F(l2_sqrt_test, l2_inplace_Sqrt_test_uint64) {
  auto selfDesc = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceSqrt, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}