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

#include "math/reduce_all/op_api/aclnn_all.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_all_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "all_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "all_test TearDown" << endl;
  }
};

// 测试一维tensor, dim越界
TEST_F(l2_all_test, case_2_dim_out_range) {
  auto tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{1});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试二维tensor
TEST_F(l2_all_test, case_3_dim_2) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试二维tensor
TEST_F(l2_all_test, case_4_dim_2) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 1}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{1});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试二维tensor, keepdim为false
TEST_F(l2_all_test, case_5_dim_2) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试二维tensor
TEST_F(l2_all_test, case_6_dim_2) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{1});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试二维tensor
TEST_F(l2_all_test, case_7_dim_2) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试二维tensor
TEST_F(l2_all_test, case_8_dim_2) {
  auto tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_9_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_10_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{1});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_11_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{2});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_12_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{-1});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_13_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{-2});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_14_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{-3});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor,dim越界
TEST_F(l2_all_test, case_15_dim_out_range) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{3});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试三维tensor,dim越界
TEST_F(l2_all_test, case_16_out_range) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{-4});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试三维tensor
TEST_F(l2_all_test, case_17_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_18_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 1, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{1});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_19_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{2});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_20_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{-1});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_21_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 1, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{-2});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_all_test, case_22_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{-3});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 测试三维tensor
TEST_F(l2_all_test, case_23_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{3});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试三维tensor,dim越界
TEST_F(l2_all_test, case_24_out_range) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{-4});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试三维tensor,dim重复
TEST_F(l2_all_test, case_34_duplicated_dim) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 1, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试三维tensor
TEST_F(l2_all_test, case_25_dim_3) {
  auto tensor_desc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 空指针
TEST_F(l2_all_test, case_26_self_nullptr) {
  auto tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;

  auto ut = OP_API_UT(aclnnAll, INPUT((aclTensor*)nullptr, dim_desc, keepdim), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针
TEST_F(l2_all_test, case_27_dim_nullptr) {
  auto tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;

  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, (aclIntArray*)nullptr, keepdim), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针
TEST_F(l2_all_test, case_28_nullptr) {
  auto tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;

  auto ut = OP_API_UT(aclnnAll, INPUT((aclTensor*)nullptr, (aclIntArray*)nullptr, keepdim), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试非连续支持
TEST_F(l2_all_test, case_29_NonContiguous) {
  auto tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
  auto out_tensor_desc = TensorDesc({1, 4}, ACL_BOOL, ACL_FORMAT_ND, {4, 1}, 0, {4, 1});  //

  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;

  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  // ut.TestPrecision();
}

// 测试超过8维tensor
TEST_F(l2_all_test, case_30_dim_9) {
  auto tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = true;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空进非空出
TEST_F(l2_all_test, case_31_empty_self) {
  auto tensor_desc = TensorDesc({2, 0, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{1});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  // ut.TestPrecision();
}

// 空进空出
TEST_F(l2_all_test, case_32_empty_self_out) {
  auto tensor_desc = TensorDesc({2, 0, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({0, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{0});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  // ut.TestPrecision();
}

// 空dim
TEST_F(l2_all_test, case_33_empty_dim) {
  auto tensor_desc = TensorDesc({2, 0, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim_desc = IntArrayDesc(vector<int64_t>{});
  bool keepdim = false;
  auto ut = OP_API_UT(aclnnAll, INPUT(tensor_desc, dim_desc, keepdim), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
