/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.s
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "level2/aclnn_copy.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_copy_test : public testing::Test {
protected:
 static void SetUpTestCase() { cout << "copy_test SetUp" << endl; }

 static void TearDownTestCase() { cout << "copy_test TearDown" << endl; }
};

// 正常路径，空tensor
TEST_F(l2_copy_test, case_empty) {
  auto self_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto src_desc = TensorDesc(self_desc);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径: int8 拷贝到 int8
TEST_F(l2_copy_test, case_int8_to_int8) {
  auto self_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: int16 拷贝到 int16
TEST_F(l2_copy_test, case_int16_to_int16) {
  auto self_desc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: int32 拷贝到 int32
TEST_F(l2_copy_test, case_int32_to_int32) {
  auto self_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: float 拷贝到 float
TEST_F(l2_copy_test, case_float_to_float) {
  auto self_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: float16 拷贝到 float16
TEST_F(l2_copy_test, case_float16_to_float16) {
  auto self_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: bool 拷贝到 bool
TEST_F(l2_copy_test, case_bool_to_bool) {
  auto self_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: double 拷贝到 float16
TEST_F(l2_copy_test, case_double_to_float16) {
  auto self_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: float 拷贝到 int32
TEST_F(l2_copy_test, case_float_to_int32) {
  auto self_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: double 拷贝到 complex64
TEST_F(l2_copy_test, case_double_to_complex64) {
  auto self_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: uint32 拷贝到 uint16
TEST_F(l2_copy_test, case_uint32_to_uint16) {
  auto self_desc = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径: uint32 拷贝到 uint64
TEST_F(l2_copy_test, case_uint32_to_uint64) {
  auto self_desc = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径: broadcast情况, float32 拷贝到 float32
TEST_F(l2_copy_test, case_float32_to_float32_broadcast) {
  auto self_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: broadcast情况, int32 拷贝到 int32
TEST_F(l2_copy_test, case_int32_to_int32_broadcast) {
  auto self_desc = TensorDesc({4, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({1, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: broadcast情况, double 拷贝到 double
TEST_F(l2_copy_test, case_double_to_double_broadcast) {
  auto self_desc = TensorDesc({4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({1, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: broadcast情况, float16 拷贝到 int32
TEST_F(l2_copy_test, case_float16_to_int32_broadcast) {
  auto self_desc = TensorDesc({4, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({1, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: broadcast情况, float32 拷贝到 int32
TEST_F(l2_copy_test, case_float32_to_int32_broadcast) {
  auto self_desc = TensorDesc({4, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: broadcast情况, float32 拷贝到 double
TEST_F(l2_copy_test, case_float32_to_double_broadcast) {
  auto self_desc = TensorDesc({4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: broadcast情况, int64 拷贝到 int8
TEST_F(l2_copy_test, case_int64_to_int8_broadcast) {
  auto self_desc = TensorDesc({4, 5}, ACL_INT8, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({1, 5}, ACL_INT64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径: broadcast情况, float 拷贝到 bool
TEST_F(l2_copy_test, case_float32_to_bool_broadcast) {
  auto self_desc = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// 正常路径:  bfloat16 拷贝到 float
TEST_F(l2_copy_test, case_float_to_bfloat16) {
  auto self_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径:  float 拷贝到 bfloat16
TEST_F(l2_copy_test, case_bfloat16_to_float) {
  auto self_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径: 复数拷贝到float
TEST_F(l2_copy_test, case_complex64_to_float) {
  auto self_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常路径: 数据维度大于9维
TEST_F(l2_copy_test, case_invalid_shape_9_dims) {

  auto self_desc = TensorDesc({5,4,6,1,1,1,1,1,1}, ACL_BOOL, ACL_FORMAT_ND);
  auto src_desc = TensorDesc({5,4,6,1,1,1,1,1,1}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceCopy, INPUT(self_desc, src_desc), OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
