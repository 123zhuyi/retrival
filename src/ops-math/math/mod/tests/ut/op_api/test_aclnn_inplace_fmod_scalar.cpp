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

#include "../../../op_api/aclnn_fmod_scalar.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_inplace_fmod_scalar_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "inplace_fmod_scalar_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "inplace_fmod_scalar_test TearDown" << endl; }
};

TEST_F(l2_inplace_fmod_scalar_test, aclnnInplaceFmodScalar_10_10_float_nd_3) {
  // input tensor
  const vector<int64_t>& selfShape = {10, 10};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // scalar
  double scalarValue = 3.0;
  aclDataType scalarDtype = ACL_FLOAT;
  // output
  const vector<int64_t>& outShape = {10, 10};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 100);
  auto scalarDesc = ScalarDesc(scalarValue, scalarDtype);

  auto ut = OP_API_UT(aclnnInplaceFmodScalar, INPUT(selfTensorDesc, scalarDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fmod_scalar_test, aclnnInplaceFmodScalar_20_int32_nd_7) {
  // input tensor
  const vector<int64_t>& selfShape = {20};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // scalar
  int32_t scalarValue = 7;
  aclDataType scalarDtype = ACL_INT32;
  // output
  const vector<int64_t>& outShape = {20};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 100);
  auto scalarDesc = ScalarDesc(scalarValue);

  auto ut = OP_API_UT(aclnnInplaceFmodScalar, INPUT(selfTensorDesc, scalarDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fmod_scalar_test, aclnnInplaceFmodScalar_20_int64_nd_7) {
  // input tensor
  const vector<int64_t>& selfShape = {20};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // scalar
  int64_t scalarValue = 7;
  aclDataType scalarDtype = ACL_INT64;
  // output
  const vector<int64_t>& outShape = {20};
  aclDataType outDtype = ACL_INT64;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 100);
  auto scalarDesc = ScalarDesc(scalarValue);

  auto ut = OP_API_UT(aclnnInplaceFmodScalar, INPUT(selfTensorDesc, scalarDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}