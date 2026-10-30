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
#include "aclnn_cosh.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_cosh_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_cosh_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "l2_cosh_test TearDown" << std::endl; }
};

// self为空指针
TEST_F(l2_cosh_test, l2_cosh_test_001) {
  auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT((aclTensor*)nullptr), OUTPUT(tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// self和out的shape类型不一致
TEST_F(l2_cosh_test, l2_cosh_test_002) {
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self和out的shape类型不一致
TEST_F(l2_cosh_test, l2_cosh_test_003) {
  auto selfDesc = TensorDesc({2, 3, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_cosh_test, l2_cosh_test_004) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32
TEST_F(l2_cosh_test, l2_cosh_test_005) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

}

// 正常路径，bfloat16
TEST_F(l2_cosh_test, ascend910B2_l2_cosh_test_bf16) {
  auto selfDesc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

}

//正常路径，int32
TEST_F(l2_cosh_test, l2_cosh_test_007) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

}

// 正常路径，int8
TEST_F(l2_cosh_test, l2_cosh_test_008) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

}

// 正常路径，int16
TEST_F(l2_cosh_test, l2_cosh_test_009) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

}

// 输入输出shape不同
TEST_F(l2_cosh_test, l2_cosh_test_010) {
  auto selfDesc = TensorDesc({2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

//正常路径，float32，9维张量
TEST_F(l2_cosh_test, l2_cosh_test_011) {
  auto selfDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

//正常路径，float32，8维张量
TEST_F(l2_cosh_test, l2_cosh_test_012) {
  auto selfDesc = TensorDesc({5, 2, 3, 4, 6, 1, 7, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({5, 2, 3, 4, 6, 1, 7, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

//0维场景
TEST_F(l2_cosh_test, l2_cosh_test_013) {
  auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，bool
TEST_F(l2_cosh_test, l2_cosh_test_014) {
  auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

}

// 非连续场景
TEST_F(l2_cosh_test, case_discontiguous_float)
{
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-20, 20);
  auto outDesc = TensorDesc(selfDesc);
  auto ut = OP_API_UT(aclnnCosh, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

}

TEST_F(l2_cosh_test, l2_inplacecosh_not_supported_dtype) {
  auto selfDesc = TensorDesc({2, 4, 4}, ACL_INT8, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCosh, INPUT(selfDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}