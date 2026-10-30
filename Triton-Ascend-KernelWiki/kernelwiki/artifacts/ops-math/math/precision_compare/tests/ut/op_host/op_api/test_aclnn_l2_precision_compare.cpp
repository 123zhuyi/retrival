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

#include "math/precision_compare/op_host/op_api/aclnn_precision_compare.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2_precision_compare_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "tensor_precision_compare_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "tensor_precision_compare_test TearDown" << endl; }
};

TEST_F(l2_precision_compare_test, ascend910B2_test_input_benchmark_is_null_failed) {
  auto tensor_benchmark = nullptr;
  auto tensor_realdata = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});
  auto out_tensor_desc = DescToAclContainer(TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND));
  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  EXPECT_EQ(workspace_size, 5UL);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_input_realdata_is_null_failed) {
  auto tensor_benchmark = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 5, 6, 7, 8});
  auto tensor_realdata = nullptr;
  auto out_tensor_desc = DescToAclContainer(TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND));
  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  EXPECT_EQ(workspace_size, 5UL);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_out_is_null_failed) {
  auto tensor_benchmark = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto tensor_realdata = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto out_tensor_desc = nullptr;
  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  EXPECT_EQ(workspace_size, 5UL);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_input_dtype_not_same_failed) {
  auto tensor_benchmark = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto tensor_realdata = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto out_tensor_desc = DescToAclContainer(TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND));

  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 5UL);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_input_benchmark_dtype_invalid) {
  auto tensor_benchmark = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto tensor_realdata = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto out_tensor_desc = DescToAclContainer(TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND));

  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 5UL);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_output_dtype_invalid) {
  auto tensor_benchmark = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto tensor_realdata = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto out_tensor_desc = DescToAclContainer(TensorDesc({}, ACL_UINT8, ACL_FORMAT_ND));

  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 5UL);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_input_shape_not_equal_failed) {
  auto tensor_benchmark = TensorDesc({2,2}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto tensor_realdata = TensorDesc({1,2}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto out_tensor_desc = DescToAclContainer(TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND));

  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  EXPECT_EQ(workspace_size, 0UL);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_input_benchmark_dimnum_invalid) {
  auto tensor_benchmark = TensorDesc({2,2,2,2,2,2,2,2,2}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto tensor_realdata = TensorDesc({2,2,2,2,2,2,2,2,2}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto out_tensor_desc = DescToAclContainer(TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND));
  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_out_shape_invalid) {
  auto tensor_benchmark = TensorDesc({2,2}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto tensor_realdata = TensorDesc({2,2}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto out_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT32, ACL_FORMAT_NCDHW).Precision(1, 2);
  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_input_empty_success) {
  auto tensor_benchmark = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto tensor_realdata = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto out_tensor_desc = DescToAclContainer(TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND));

  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  EXPECT_EQ(workspace_size, 0UL);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_benchmark_realdata_equal_success) {
  auto tensor_benchmark = TensorDesc({2,3}, ACL_FLOAT,ACL_FORMAT_ND)
  .ValueRange(-1, 1)
  .Value(vector<float>{-6.75468990e-01,7.37894118e-01,-7.84537017e-01,9.64608252e-01,9.09311235e-01,-9.51533616e-01});
  auto tensor_realdata = TensorDesc({2,3},ACL_FLOAT,ACL_FORMAT_ND)
  .ValueRange(-1, 1)
  .Value(vector<float>{-6.75468990e-01,7.37894118e-01,-7.84537017e-01,9.64608252e-01,9.09311235e-01,-9.51533616e-01});

  auto outDesc = TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_precision_compare_test, ascend910B2_test_benchmark_realdata_not_equal_501) {
  auto tensor_benchmark = TensorDesc({2,3}, ACL_FLOAT,ACL_FORMAT_ND)
  .ValueRange(-1, 1)
  .Value(vector<float>{-6.75468990e-01,7.37894118e-01,-7.84537017e-01,9.64608252e-01,9.09311235e-01,-9.51533616e-01});
  auto tensor_realdata = TensorDesc({2,3},ACL_FLOAT,ACL_FORMAT_ND)
  .ValueRange(-1, 1)
  .Value(vector<float>{-6.75468990e-01,7.39894118e-01,-7.84537017e-01,9.64608252e-01,9.09311235e-01,-9.51533616e-01});

  auto outDesc = TensorDesc({}, ACL_UINT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnPrecisionCompare, INPUT(tensor_benchmark, tensor_realdata), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}