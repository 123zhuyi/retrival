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

#include "aclnn_argmax.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"


using namespace std;

class l2_argmax_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_argmax_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_argmax_test TearDown" << endl;
    }
};

TEST_F(l2_argmax_test, aclnnArgMax_3_4_float_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}


TEST_F(l2_argmax_test, ascend310P_aclnnArgMax_3_4_6_float16_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,4,6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({4,6}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_16_40_10_float16_nd_1_nd) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({16,40,10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({16,10}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();  // comment bcz of timeout in model tests (330454 ms)
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_float_nd_1_nd) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({16,100,2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({16,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_double_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({100,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_256_40_10_double_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({256,40,10}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({40,10}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_int8_nd_1_nd) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({512,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_int16_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({100,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_argmax_test, ascend910B2_aclnnArgMax_512_100_2_int32_nd_1_nd) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({512,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_int64_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({100,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_uint8_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({100,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_int8_nd_1_nd_bool_dtype) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({512,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_int8_nd_1_nd_uint32_dtype) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({512,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_int8_nd_1_nd_uint64_dtype) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,100,2}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({512,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmax_test, aclnnArgMax_512_100_2_float_nd_1_nd_transpose) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,5,7,6}, ACL_FLOAT, ACL_FORMAT_ND,{210,42,1,7},0,{3,5,6,7}).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({5,7,6}, ACL_INT64, ACL_FORMAT_ND,{42,1,7},0,{5,6,7}).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_empty_tensor_int8_nd_1_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({512,0,2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({512,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmax_test, aclnnArgMax_8_dim_int64_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({2,2,2,2,2,2,2,2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({2,2,2,2,2,2,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // ut.TestPrecision();
}

TEST_F(l2_argmax_test, aclnnArgMax_1_int64_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmax_test, aclnnArgMax_8_dim_int64_with_dim_7) {
  const int64_t dim = 7;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({2,2,2,2,2,2,2,2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({2,2,2,2,2,2,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnArgMax, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}