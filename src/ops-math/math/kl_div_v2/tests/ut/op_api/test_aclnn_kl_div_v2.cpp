/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "math/kl_div_v2/op_api/aclnn_kl_div.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_kl_div_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "kl_div SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "kl_div TearDown" << endl;
  }
};

// 非法数据类型double
TEST_F(l2_kl_div_test, case_003_double) {
  auto selfDesc = TensorDesc({3, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({3, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t reduction = 0;
  bool logTarget = false;

  auto outDesc = TensorDesc({3, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法数据类型uint8
TEST_F(l2_kl_div_test, case_004_uint8) {
  auto selfDesc = TensorDesc({5, 3, 5}, ACL_UINT8, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({5, 3, 5}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t reduction = 0;
  bool logTarget = false;

  auto outDesc = TensorDesc({5, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法数据类型int8
TEST_F(l2_kl_div_test, case_005_int8) {
  auto selfDesc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t reduction = 0;
  bool logTarget = false;

  auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法数据类型int16
TEST_F(l2_kl_div_test, case_006_int16) {
  auto selfDesc = TensorDesc({3, 3, 5}, ACL_INT16, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({3, 3, 5}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t reduction = 0;
  bool logTarget = false;

  auto outDesc = TensorDesc({3, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法数据类型int32
TEST_F(l2_kl_div_test, case_007_int32) {
  auto selfDesc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t reduction = 0;
  bool logTarget = false;

  auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法数据类型int64
TEST_F(l2_kl_div_test, case_008_int64) {
  auto selfDesc = TensorDesc({3, 5}, ACL_INT64, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({3, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t reduction = 0;
  bool logTarget = false;

  auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法数据类型bool
TEST_F(l2_kl_div_test, case_009_bool) {
  auto selfDesc = TensorDesc({3, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({3, 5}, ACL_BOOL, ACL_FORMAT_ND);
  int64_t reduction = 0;
  bool logTarget = false;

  auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法数据类型complex64
TEST_F(l2_kl_div_test, case_010_complex64) {
  auto selfDesc = TensorDesc({3, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({3, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
  int64_t reduction = 0;
  bool logTarget = false;

  auto outDesc = TensorDesc({3, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空指针
TEST_F(l2_kl_div_test, case_011_nullptr) {
  auto selfDesc = TensorDesc({10, 3, 5, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 3, 5, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto outDesc = TensorDesc({10, 3, 5, 24}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;
  bool logTarget = true;

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  auto ut_1 = OP_API_UT(aclnnKlDiv, INPUT(nullptr, targetDesc, reduction, logTarget), OUTPUT(outDesc));
  aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_2 = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, nullptr, reduction, logTarget), OUTPUT(outDesc));
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_3 = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(nullptr));
  aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空Tensor
TEST_F(l2_kl_div_test, case_012_empty) {
  auto selfDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t reduction = 0;
  bool logTarget = true;

  auto outDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 维度大于8
TEST_F(l2_kl_div_test, case_013_max_dim) {
  auto selfDesc = TensorDesc({10, 24, 3, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 24, 3, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t reduction = 0;
  bool logTarget = true;

  auto outDesc = TensorDesc({10, 24, 3, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self与target不能做broadcast
TEST_F(l2_kl_div_test, case_015_broadcast_failed) {
  auto selfDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 4, 6, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t reduction = 0;
  bool logTarget = true;

  auto outDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out与广播形状不一致
TEST_F(l2_kl_div_test, case_016_out_not_equal_broadcast) {
  auto selfDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t reduction = 0;
  bool logTarget = true;

  auto outDesc = TensorDesc({3, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// reduction参数为mean
TEST_F(l2_kl_div_test, case_017_reduction_mean) {
  auto selfDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = 1;
  bool logTarget = false;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// reduction参数为sum
TEST_F(l2_kl_div_test, case_018_reduction_sum) {
  auto selfDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(0, 1);
  int64_t reduction = 2;
  bool logTarget = false;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空Tensor, reduction == 1
TEST_F(l2_kl_div_test, case_022_empty_sum) {
  auto selfDesc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t reduction = 1;
  bool logTarget = false;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空Tensor, reduction == 2
TEST_F(l2_kl_div_test, case_023_empty_mean) {
  auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t reduction = 2;
  bool logTarget = false;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 1D tensor mean
TEST_F(l2_kl_div_test, case_024_1D_mean) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t reduction = 1;
  bool logTarget = false;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 0D tensor mean
TEST_F(l2_kl_div_test, case_025_0D_mean) {
  auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t reduction = 1;
  bool logTarget = false;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnKlDiv, INPUT(selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}