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

#include "../../../op_api/aclnn_fmod_tensor.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_fmod_tensor_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "fmod_tensor_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "fmod_tensor_test TearDown" << endl; }
};

TEST_F(l2_fmod_tensor_test, aclnnFmodTensor_10_10_float_nd_10_10_float_nd) {
  // left input
  const vector<int64_t>& selfShape = {10, 10};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {10, 10};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {10, 10};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 100);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 10);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnFmodTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fmod_tensor_test, aclnnFmodTensor_5_8_float16_nd_5_8_float16_nd) {
  // left input
  const vector<int64_t>& selfShape = {5, 8};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {5, 8};
  aclDataType otherDtype = ACL_FLOAT16;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {5, 8};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 100);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 10);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnFmodTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fmod_tensor_test, aclnnFmodTensor_20_int32_nd_3_int32_nd) {
  // left input
  const vector<int64_t>& selfShape = {20};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {3};
  aclDataType otherDtype = ACL_INT32;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {20};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 100);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 10);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0, 0);

  auto ut = OP_API_UT(aclnnFmodTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fmod_tensor_test, aclnnFmodTensor_20_int64_nd_3_int32_nd) {
  // left input
  const vector<int64_t>& selfShape = {20};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {20};
  aclDataType otherDtype = ACL_INT64;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {20};
  aclDataType outDtype = ACL_INT64;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 100);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 10);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0, 0);

  auto ut = OP_API_UT(aclnnFmodTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
