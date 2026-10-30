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

#include "../../../op_api/aclnn_addr.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_inplace_addr_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "inplace_addr_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "inplace_addr_test TearDown" << endl;
  }
};

// *** tensor rank range ***
// self dims is not 2 for inplaceAddr
TEST_F(l2_inplace_addr_test, case_inplace_addr_for_self_dims_not_two) {
  auto self_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnInplaceAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self shape not equal outer shape
TEST_F(l2_inplace_addr_test, case_inplace_addr_for_self_unequal_shape) {
  auto self_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec1_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto vec2_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto beta_scalar_desc = ScalarDesc(1.0f);
  auto alpha_scalar_desc = ScalarDesc(1.0f);

  auto ut = OP_API_UT(aclnnInplaceAddr,
                      INPUT(self_tensor_desc, vec1_tensor_desc, vec2_tensor_desc, beta_scalar_desc, alpha_scalar_desc),
                      OUTPUT());

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}