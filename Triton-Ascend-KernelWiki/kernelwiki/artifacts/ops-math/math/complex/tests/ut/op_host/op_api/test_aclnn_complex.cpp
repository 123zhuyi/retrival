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
#include "../../../../op_host/op_api/aclnn_complex.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class complex_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "complex_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "complex_test TearDown" << endl; }
};


TEST_F(complex_test, ascend910B2_complex_test_complex64) {
  auto tensor_real = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_imag = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_out = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnComplex, INPUT(tensor_real, tensor_imag), OUTPUT(tensor_out));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(complex_test, ascend910B2_complex_test_complex32) {
   auto tensor_real = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
   auto tensor_imag = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
   auto tensor_out = TensorDesc({2, 3}, ACL_COMPLEX32, ACL_FORMAT_ND);
   auto ut = OP_API_UT(aclnnComplex, INPUT(tensor_real, tensor_imag), OUTPUT(tensor_out));

   // SAMPLE: only test GetWorkspaceSize
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
   EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(complex_test, ascend910B2_complex_test_complex128) {
  auto tensor_real = TensorDesc({2, 3}, ACL_COMPLEX32, ACL_FORMAT_ND);
  auto tensor_imag = TensorDesc({2, 3}, ACL_COMPLEX32, ACL_FORMAT_ND);
  auto tensor_out = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnComplex, INPUT(tensor_real, tensor_imag), OUTPUT(tensor_out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(complex_test, ascend910B2_complex_test_output_check) {
  auto tensor_real = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto tensor_imag = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto tensor_out = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnComplex, INPUT(tensor_real, tensor_imag), OUTPUT(tensor_out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试空tensor
TEST_F(complex_test, ascend910B2_case_empty_tensors)
{
    auto self_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnComplex, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}