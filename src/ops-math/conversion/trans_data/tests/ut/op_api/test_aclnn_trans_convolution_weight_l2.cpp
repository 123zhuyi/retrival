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

#include "conversion/trans_data/op_api/aclnn_trans_convolution_weight.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_trans_convolution_weight_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_trans_convolution_weight_test SetUp" << endl; }
  static void TearDownTestCase() { cout << "l2_trans_convolution_weight_test TearDown" << endl; }
};

TEST_F(l2_trans_convolution_weight_test, ascend310P3_test_normal_input_FP16) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 4, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc(x1_desc);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_test_normal_input_FP32) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 4, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc({16, 4, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_test_wrong_format) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 4, 5, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto y_desc = TensorDesc(x1_desc);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_test_wrong_dtype_input_int8) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 4, 5, 5}, ACL_INT8, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc(x1_desc);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_test_wrong_format_out) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 4, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc({16, 4, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_test_wrong_dtype_input_int8_out) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 4, 5, 5}, ACL_INT8, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc({16, 4, 5, 5}, ACL_INT8, ACL_FORMAT_NCHW);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_dim_larger_than_4) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 16, 32, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc(x1_desc);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


TEST_F(l2_trans_convolution_weight_test, ascend310P3_group_0) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 16, 32, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc(x1_desc);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 0), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_transpose_true) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 16, 32, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc(x1_desc);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, true, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_in_nullptr) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 16, 32, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc(x1_desc);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(nullptr, false, 1), OUTPUT(y_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P3_out_nullptr) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  // 使用**Desc描述host api输入输出
  auto x1_desc = TensorDesc({16, 16, 32, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto y_desc = TensorDesc(x1_desc);
  auto ut = OP_API_UT(aclnnTransConvolutionWeight, INPUT(x1_desc, false, 1), OUTPUT(nullptr));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P_normal_shape) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  aclIntArray* tensorShape = nullptr;
  vector<int64_t> tensorShapeVec = {2, 2, 32, 16};
  tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
  uint64_t weightSize = 0;
  aclnnStatus aclRet = aclnnCalculateConvolutionWeightSize(tensorShape, false, 1, ACL_FLOAT16, &weightSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P_shape_nullptr) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  uint64_t weightSize = 0;
  aclnnStatus aclRet = aclnnCalculateConvolutionWeightSize(nullptr, false, 1, ACL_FLOAT16, &weightSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P_size_nullptr) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  aclIntArray* tensorShape = nullptr;
  vector<int64_t> tensorShapeVec = {2, 2, 32, 16};
  tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
  aclnnStatus aclRet = aclnnCalculateConvolutionWeightSize(tensorShape, false, 1, ACL_FLOAT16, nullptr);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P_datatype_fp32) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  aclIntArray* tensorShape = nullptr;
  vector<int64_t> tensorShapeVec = {2, 2, 32, 16};
  tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
  uint64_t weightSize = 0;
  aclnnStatus aclRet = aclnnCalculateConvolutionWeightSize(tensorShape, false, 1, ACL_FLOAT, &weightSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P_group_0) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  aclIntArray* tensorShape = nullptr;
  vector<int64_t> tensorShapeVec = {2, 2, 32, 16};
  tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
  uint64_t weightSize = 0;
  aclnnStatus aclRet = aclnnCalculateConvolutionWeightSize(tensorShape, false, 0, ACL_FLOAT16, &weightSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P_wrong_dim) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  aclIntArray* tensorShape = nullptr;
  vector<int64_t> tensorShapeVec = {2, 32, 16};
  tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
  uint64_t weightSize = 0;
  aclnnStatus aclRet = aclnnCalculateConvolutionWeightSize(tensorShape, false, 0, ACL_FLOAT16, &weightSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_convolution_weight_test, ascend310P_transpose_true) {
  SetPlatformNpuArch(NpuArch::DAV_2002);
  aclIntArray* tensorShape = nullptr;
  vector<int64_t> tensorShapeVec = {2, 2, 32, 16};
  tensorShape = aclCreateIntArray(tensorShapeVec.data(), tensorShapeVec.size());
  uint64_t weightSize = 0;
  aclnnStatus aclRet = aclnnCalculateConvolutionWeightSize(tensorShape, true, 1, ACL_FLOAT16, &weightSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}