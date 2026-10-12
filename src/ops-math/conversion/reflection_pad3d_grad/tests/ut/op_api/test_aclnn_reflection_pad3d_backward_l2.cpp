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

#include "../../../op_api/aclnn_reflection_pad3d_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class reflection_pad3d_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "reflection_pad3d_backward_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "reflection_pad3d_backward_test TearDown" << endl; }
};

// 空tensor
TEST_F(reflection_pad3d_backward_test, case_1) {
  auto grad_output_tensor_desc = TensorDesc({0, 6, 6, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto input_tensor_desc = TensorDesc({0, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
  auto grad_input_desc = TensorDesc({0, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull gradOutput input padding
TEST_F(reflection_pad3d_backward_test, case_2) {
  auto grad_output_tensor_desc = TensorDesc({2, 8, 7, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({2, 4, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({2, 4, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(nullptr, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_2 = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, nullptr, padding_desc),
                        OUTPUT(grad_input_desc));
  workspace_size = 0;
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_3 = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, nullptr),
                        OUTPUT(grad_input_desc));
  workspace_size = 0;
  aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull gradInput
TEST_F(reflection_pad3d_backward_test, case_3) {
  auto grad_output_tensor_desc = TensorDesc({2, 8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({2, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({2, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(nullptr));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckShape diffrent shape of input and gradInput
TEST_F(reflection_pad3d_backward_test, case_4)
{
  auto grad_output_tensor_desc = TensorDesc({2, 8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({2, 5, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({2, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape padding dim
TEST_F(reflection_pad3d_backward_test, case_5)
{
  auto grad_output_tensor_desc = TensorDesc({2, 8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({2, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2});
  auto grad_input_desc = TensorDesc({2, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape input dim
TEST_F(reflection_pad3d_backward_test, case_6)
{
  auto grad_output_tensor_desc = TensorDesc({8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape diffrent dim of input and gradOutput
TEST_F(reflection_pad3d_backward_test, case_7)
{
  auto grad_output_tensor_desc = TensorDesc({8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat diffrent format
TEST_F(reflection_pad3d_backward_test, case_8)
{
  auto grad_output_tensor_desc = TensorDesc({1, 1, 8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_UNDEFINED);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype support
TEST_F(reflection_pad3d_backward_test, case_9)
{
  vector<aclDataType> ValidList = {
      ACL_FLOAT16,
      ACL_FLOAT,
      ACL_DOUBLE,
      ACL_COMPLEX64,
      ACL_COMPLEX128};

  int length = ValidList.size();
  for (int i = 0; i < length; i++) {
    auto grad_output_tensor_desc = TensorDesc({1, 1, 8, 8, 8}, ValidList[i], ACL_FORMAT_ND).ValueRange(1, 1);
    auto input_tensor_desc = TensorDesc({1, 1, 4, 4, 4}, ValidList[i], ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
    auto grad_input_desc = TensorDesc({1, 1, 4, 4, 4}, ValidList[i], ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                        OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

// CheckDtype not support
TEST_F(reflection_pad3d_backward_test, case_10)
{
  auto grad_output_tensor_desc = TensorDesc({1, 1, 8, 8, 8}, ACL_INT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_INT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype diffrent dtype of gradOutput and gradInput
TEST_F(reflection_pad3d_backward_test, case_11)
{
  auto grad_output_tensor_desc = TensorDesc({1, 1, 8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype diffrent dtype of input and gradOutput
TEST_F(reflection_pad3d_backward_test, case_12)
{
  auto grad_output_tensor_desc = TensorDesc({1, 1, 8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape padding dim value
TEST_F(reflection_pad3d_backward_test, case_13)
{
  auto grad_output_tensor_desc = TensorDesc({1, 8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3, 4, 4});
  auto grad_input_desc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape gradOutput shape
TEST_F(reflection_pad3d_backward_test, case_14)
{
  auto grad_output_tensor_desc = TensorDesc({1, 5, 5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
  auto input_tensor_desc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
  auto grad_input_desc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(reflection_pad3d_backward_test, case_15) {
  auto grad_output_tensor_desc = TensorDesc({3, 4, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto input_tensor_desc = TensorDesc({3, 0, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2});
  auto grad_input_desc = TensorDesc({3, 0, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnReflectionPad3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc, padding_desc),
                      OUTPUT(grad_input_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
